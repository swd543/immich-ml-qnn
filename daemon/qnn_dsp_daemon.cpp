// qnn_dsp_daemon — resident QNN inference server for Qualcomm Hexagon cDSP (HTP backend).
//
// Loads QNN context binaries at startup and serves them over plain HTTP:
//   GET  /health           -> JSON status
//   POST /infer/<model>    -> raw float32 NCHW input  =>  raw float32 output
//
// The context binaries used here are INT8 (uFxp_8) end to end: the daemon
// quantizes the float32 request body into the input encoding, runs the graph
// on the DSP, and dequantizes the int8 output back to float32 — exactly what
// qnn-net-run does internally (verified bit-identical on this board).
//
// Tensor ground truth (from qairt-dlc-info on the quantized DLCs):
//   clip    graph "clipr37":   in "image"    [1,3,224,224] uFxp_8 scale 0.015443762764 off -116
//                               out "embedding" [1,512]   uFxp_8 scale 0.003277973738 off -207
//   arcface graph "arcface37": in "input_1" [1,3,112,112] uFxp_8 scale 0.006889658049 off -136
//                               out "_683"       [1,512]   uFxp_8 scale 0.015603637323 off -123
//
// Build:
//   g++ -O2 -std=c++17 -Wall -I<qairst-include> qnn_dsp_daemon.cpp -o qnn_dsp_daemon -ldl -lpthread
//
// The HTP backend locates the DSP skel (libQnnHtpV68Skel.so) via the current
// working directory, so the process must run with cwd = the runtime directory
// containing libQnnHtp.so / skel / stub.

#include <QNN/QnnInterface.h>

#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cmath>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

// ---------------------------------------------------------------- logging --
static void logmsg(const char* lvl, const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  fprintf(stderr, "[qnn-dsp-daemon %s] %s\n", lvl, buf);
  fflush(stderr);
}

// ------------------------------------------------------- model descriptors --
// Per-output tensor spec (used for multi-output graphs such as SCRFD, which
// emits score/box/kps tensors for each FPN level).
struct OutSpec {
  const char* name;
  uint32_t dims[4];   // rank 2 here: {d0, d1, 1, 1}
  float scale;
  int32_t offset;
  uint32_t id;        // tensor id from context binary metadata
};

struct ModelSpec {
  const char* key;          // endpoint name
  const char* graphName;    // graph name inside the context binary
  const char* inName;       // graph input tensor name
  const char* outName;      // graph output tensor name (single-output models)
  uint32_t inDims[4];
  uint32_t outDims[2];
  float inScale;
  int32_t inOffset;
  float outScale;
  int32_t outOffset;
  uint32_t inId;   // tensor id from context binary metadata
  uint32_t outId;
  const OutSpec* outs;     // multi-output list (null for single-output models)
  int numOuts;             // 0 = single output (legacy fields above)
};

struct Model {
  const ModelSpec* spec = nullptr;
  std::string contextPath;
  Qnn_ContextHandle_t ctx = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  std::vector<uint32_t> inDims;   // stable storage for tensor dims
  Qnn_Tensor_t inTensor{};
  std::vector<uint32_t> outDimsList;  // per-output dims (stable storage)
  std::vector<Qnn_Tensor_t> outTensors;
  std::vector<std::vector<uint8_t>> outBufs;  // int8 output per tensor (1 byte/elem)
  std::vector<float> outScales;
  std::vector<int32_t> outOffsets;
  std::vector<size_t> outElemsList;
  size_t outElemsTotal = 0;
  std::vector<uint8_t> inBuf;     // int8 input  (1 byte/elem)
  uint64_t runs = 0;
  double lastMs = 0.0;
};

// ---------------------------------------------------------------- globals --
static std::atomic<bool> g_running{true};
// QNN_API_VERSION 2.28: QnnInterface_t wraps the implementation struct in an
// anonymous union (member v2_28); QNNFN() accesses it directly.
static QnnInterface_t g_if{};
#define QNNFN(name) (g_if.v2_28.name)
static Qnn_BackendHandle_t g_backend = nullptr;
static Qnn_DeviceHandle_t g_device = nullptr;
static std::vector<Model> g_models;
static std::mutex g_execMu;       // single NSP: serialize graph execution
static uint64_t g_totalRuns = 0, g_totalErrors = 0;

static void onSignal(int) { g_running = false; }

// --------------------------------------------------------- tensor builder --
// The context binaries declare their I/O tensors with dataFormat 1032 (NCHW layout
// flags as written by the 2.37 converter with preserve_io layout). The backend
// validates against this value, so it must be mirrored exactly.
static uint32_t gTensorDataFormat = 1032;
static bool gNoDevice = false;
// NB: `dims` must point at caller-owned storage that outlives the tensor —
// the QNN backend dereferences dimensions during graphExecute.
static Qnn_Tensor_t makeTensor(const char* name,
                               Qnn_TensorType_t type,
                               const uint32_t* dims,
                               uint32_t rank,
                               float scale,
                               int32_t offset,
                               uint8_t* buf,
                               size_t bytes,
                               uint32_t id) {
  Qnn_Tensor_t t;
  memset(&t, 0, sizeof(t));
  t.version = QNN_TENSOR_VERSION_1;
  t.v1.id = id;
  t.v1.name = name;
  t.v1.type = type;
  t.v1.dataFormat = gTensorDataFormat;
  t.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
  t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  t.v1.quantizeParams.scaleOffsetEncoding.scale = scale;
  t.v1.quantizeParams.scaleOffsetEncoding.offset = offset;
  t.v1.rank = rank;
  t.v1.dimensions = const_cast<uint32_t*>(dims);
  t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf.data = buf;
  t.v1.clientBuf.dataSize = static_cast<uint32_t>(bytes);
  return t;
}

static size_t vol(const std::vector<uint32_t>& d) {
  size_t v = 1;
  for (uint32_t x : d) v *= x;
  return v;
}

// ------------------------------------------------------------ model setup --
static bool loadModel(const ModelSpec* spec, const std::string& path) {
  Model m;
  m.spec = spec;
  m.contextPath = path;

  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    logmsg("ERR", "cannot open context binary: %s", path.c_str());
    return false;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> file(static_cast<size_t>(sz));
  if (fread(file.data(), 1, file.size(), f) != file.size()) {
    logmsg("ERR", "short read on %s", path.c_str());
    fclose(f);
    return false;
  }
  fclose(f);

  {
    Qnn_ErrorHandle_t cerra = QNNFN(contextCreateFromBinary)(g_backend, g_device, nullptr, file.data(),
                                                             file.size(), &m.ctx, nullptr);
    if (cerra != QNN_SUCCESS) {
      logmsg("ERR", "contextCreateFromBinary failed for %s: 0x%x", path.c_str(), (unsigned)cerra);
      return false;
    }
  }
  if (!QNNFN(graphRetrieve) || QNNFN(graphRetrieve)(m.ctx, spec->graphName, &m.graph) != QNN_SUCCESS) {
    logmsg("ERR", "graphRetrieve(%s) failed for %s — is the graph name right?", spec->graphName,
           path.c_str());
    // Do not leak the context we just created.
    if (m.ctx && QNNFN(contextFree)) QNNFN(contextFree)(m.ctx, nullptr);
    return false;
  }

  m.inDims.assign(spec->inDims, spec->inDims + 4);
  size_t inElems = vol(m.inDims);
  m.inBuf.resize(inElems);
  m.inTensor = makeTensor(spec->inName, QNN_TENSOR_TYPE_APP_WRITE, m.inDims.data(), 4,
                          spec->inScale, spec->inOffset, m.inBuf.data(), inElems, spec->inId);

  // Output tensors: a single-output model is the 1-element case of the same
  // path (bit-identical response layout to the original single-output code).
  int nOut = spec->numOuts > 0 ? spec->numOuts : 1;
  m.outTensors.resize(static_cast<size_t>(nOut));
  m.outBufs.resize(static_cast<size_t>(nOut));
  m.outScales.resize(static_cast<size_t>(nOut));
  m.outOffsets.resize(static_cast<size_t>(nOut));
  m.outElemsList.resize(static_cast<size_t>(nOut));
  // Rank-2 outputs: dims live in outDimsList at [2*i, 2*i+2) — stable for the
  // model's lifetime (reserve() up front so push_back never relocates), so the
  // tensors' dimensions pointers stay valid.
  m.outDimsList.reserve(static_cast<size_t>(nOut) * 2);
  for (int i = 0; i < nOut; ++i) {
    const uint32_t* od;
    if (spec->outs && spec->numOuts > 0) {
      m.outDimsList.push_back(spec->outs[i].dims[0]);
      m.outDimsList.push_back(spec->outs[i].dims[1]);
    } else {
      m.outDimsList.push_back(spec->outDims[0]);
      m.outDimsList.push_back(spec->outDims[1]);
    }
    od = &m.outDimsList.back() - 1;
    size_t oelems = static_cast<size_t>(od[0]) * od[1];
    m.outElemsList[static_cast<size_t>(i)] = oelems;
    m.outBufs[static_cast<size_t>(i)].resize(oelems);
    const char* oname = spec->outs && spec->numOuts > 0 ? spec->outs[i].name : spec->outName;
    float oscale = spec->outs && spec->numOuts > 0 ? spec->outs[i].scale : spec->outScale;
    int32_t ooff = spec->outs && spec->numOuts > 0 ? spec->outs[i].offset : spec->outOffset;
    uint32_t oid = spec->outs && spec->numOuts > 0 ? spec->outs[i].id : spec->outId;
    m.outScales[static_cast<size_t>(i)] = oscale;
    m.outOffsets[static_cast<size_t>(i)] = ooff;
    m.outTensors[static_cast<size_t>(i)] =
        makeTensor(oname, QNN_TENSOR_TYPE_APP_READ, od, 2, oscale, ooff,
                   m.outBufs[static_cast<size_t>(i)].data(), oelems, oid);
  }
  for (size_t e : m.outElemsList) m.outElemsTotal += e;

  char outLog[1024] = "";
  size_t p = 0;
  for (int i = 0; i < nOut; ++i) {
    const char* oname = spec->outs ? spec->outs[i].name : spec->outName;
    p += snprintf(outLog + p, sizeof(outLog) - p, "%s%s [%s]", i ? ", " : "", oname,
                  i ? "" : "");
  }
  logmsg("INFO", "loaded model '%s' (%s): graph '%s', in %s [%ux%ux%ux%u], outs=%d total_elems=%zu",
         spec->key, path.c_str(), spec->graphName, spec->inName, m.inDims[0], m.inDims[1],
         m.inDims[2], m.inDims[3], nOut, m.outElemsTotal);
  logmsg("DBG", "load %s: %s", spec->key, outLog);
  g_models.push_back(std::move(m));
  return true;
}

// --------------------------------------------------------------- inference --
static bool quantizeInput(Model& m, const float* inF32, size_t inElems) {
  // float32 -> uFxp_8 :  q = round(x / scale) - offset   (clamped to [0,255])
  const float inv = 1.0f / m.spec->inScale;
  for (size_t i = 0; i < inElems; ++i) {
    int32_t q = static_cast<int32_t>(lroundf(inF32[i] * inv - static_cast<float>(m.spec->inOffset)));
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    m.inBuf[i] = static_cast<uint8_t>(q);
  }
  return true;
}

static bool infer(Model& m, const uint8_t* inBuf, size_t inElems, float* outF32) {
  // keep tensor data pointers fresh (buffers are ours for the process lifetime,
  // but re-assert for clarity/safety after any vector reallocation elsewhere)
  m.inTensor.v1.clientBuf.data = const_cast<uint8_t*>(inBuf);
  for (size_t i = 0; i < m.outTensors.size(); ++i) {
    m.outTensors[i].v1.clientBuf.data = m.outBufs[i].data();
  }
  auto t0 = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(g_execMu);
  Qnn_ErrorHandle_t err = QNNFN(graphExecute)(m.graph, &m.inTensor, 1, m.outTensors.data(),
                                              static_cast<uint32_t>(m.outTensors.size()), nullptr,
                                              nullptr);
  auto t1 = std::chrono::steady_clock::now();
  m.lastMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  m.runs++;
  g_totalRuns++;
  if (err != QNN_SUCCESS) {
    logmsg("ERR", "graphExecute failed for '%s': 0x%x", m.spec->key, static_cast<unsigned>(err));
    g_totalErrors++;
    return false;
  }
  // uFxp_8 -> float32 per output :  x = (q + offset) * scale  (concatenated,
  // in spec order — the client splits the response by the known element counts)
  size_t dst = 0;
  for (size_t i = 0; i < m.outBufs.size(); ++i) {
    const float os = m.outScales[i];
    const int32_t oo = m.outOffsets[i];
    for (uint8_t q : m.outBufs[i]) {
      outF32[dst++] = (static_cast<float>(static_cast<int32_t>(q)) + static_cast<float>(oo)) * os;
    }
  }
  return true;
}

static Model* findModel(const std::string& key) {
  for (auto& m : g_models)
    if (m.spec->key == key) return &m;
  return nullptr;
}

// ------------------------------------------------------------------- http --
static void writeAll(int fd, const char* p, size_t n) {
  while (n > 0) {
    ssize_t w = write(fd, p, n);
    if (w <= 0) return;
    p += w;
    n -= static_cast<size_t>(w);
  }
}

static void sendReply(int fd, int code, const char* status, const std::string& body,
                      const char* ctype = "application/octet-stream") {
  std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status + "\r\n" +
                  "Content-Type: " + std::string(ctype) + "\r\n" +
                  "Content-Length: " + std::to_string(body.size()) + "\r\n" +
                  "Connection: close\r\n\r\n";
  writeAll(fd, h.data(), h.size());
  writeAll(fd, body.data(), body.size());
}

static std::string healthJson() {
  std::string s = "{\"status\":\"ok\",\"models\":[";
  for (size_t i = 0; i < g_models.size(); ++i) {
    Model& m = g_models[i];
    if (i) s += ",";
    s += "{\"name\":\"" + std::string(m.spec->key) + "\",\"runs\":" + std::to_string(m.runs) +
         ",\"last_ms\":" + std::to_string(m.lastMs) + "}";
  }
  s += "],\"total_runs\":" + std::to_string(g_totalRuns) + ",\"total_errors\":" +
       std::to_string(g_totalErrors) + ",\"version\":\"1.1.0\"}";
  return s;
}

static void handleConnection(int fd) {
  std::string buf;
  char tmp[4096];
  // ---- header ----
  size_t headerEnd = std::string::npos;
  while (headerEnd == std::string::npos && buf.size() < 65536) {
    ssize_t r = read(fd, tmp, sizeof(tmp));
    if (r <= 0) return;
    buf.append(tmp, static_cast<size_t>(r));
    headerEnd = buf.find("\r\n\r\n");
  }
  if (headerEnd == std::string::npos) return;

  std::string headers = buf.substr(0, headerEnd);
  std::string requestLine;
  {
    size_t eol = headers.find("\r\n");
    requestLine = headers.substr(0, eol);
  }
  std::string method, path;
  {
    size_t sp = requestLine.find(' ');
    method = requestLine.substr(0, sp);
    path = requestLine.substr(sp + 1);
    size_t q = path.find(' ');
    if (q != std::string::npos) path = path.substr(0, q);
  }
  long contentLength = 0;
  {
    std::string h = headers;
    for (auto& c : h) c = static_cast<char>(std::tolower(c));
    size_t p = h.find("content-length:");
    if (p != std::string::npos) contentLength = std::strtol(h.c_str() + p + 15, nullptr, 10);
  }

  if (method == "GET" && path == "/health") {
    sendReply(fd, 200, "OK", healthJson(), "application/json");
    return;
  }
  if (method != "POST" || path.rfind("/infer/", 0) != 0) {
    sendReply(fd, 404, "Not Found", "not found");
    return;
  }
  std::string key = path.substr(7);
  bool preQuant = false;
  {
    const std::string k = "/int8";
    if (key.size() >= k.size() && key.compare(key.size() - k.size(), k.size(), k) == 0) {
      preQuant = true;
      key.erase(key.size() - k.size());
    }
  }

  Model* m = findModel(key);
  if (!m) {
    sendReply(fd, 503, "Service Unavailable", "model not loaded: " + key);
    return;
  }
  size_t inElems = vol(m->inDims);
  size_t expectedBytes = preQuant ? inElems : inElems * sizeof(float);
  if (static_cast<size_t>(contentLength) != expectedBytes) {
    sendReply(fd, 400, "Bad Request",
              "expected " + std::to_string(expectedBytes) + " bytes, got " +
                  std::to_string(contentLength));
    return;
  }
  // ---- body ----
  // ---- body: read EXACTLY contentLength bytes (never more, never less) ----
  std::vector<char> body;
  body.reserve(static_cast<size_t>(contentLength));
  {
    std::string first = buf.substr(headerEnd + 4);
    if (first.size() > static_cast<size_t>(contentLength)) first.resize(contentLength);
    body.insert(body.end(), first.begin(), first.end());
  }
  size_t bodyReads = 0, bodyBytes = 0;
  while (body.size() < static_cast<size_t>(contentLength)) {
    size_t want = static_cast<size_t>(contentLength) - body.size();
    ssize_t r = read(fd, tmp, want < sizeof(tmp) ? want : sizeof(tmp));
    if (r <= 0) break;
    body.insert(body.end(), tmp, tmp + r);
    ++bodyReads;
    bodyBytes += static_cast<size_t>(r);
  }
  if (body.size() != static_cast<size_t>(contentLength)) {
    sendReply(fd, 400, "Bad Request", "incomplete body");
    return;
  }
  {
    // md5 of the received body (FNV-1a 64-bit — no libmd dependency)
    uint64_t h = 1469598103934665603ULL;
    for (char c : body) { h ^= static_cast<uint8_t>(c); h *= 1099511628211ULL; }
    logmsg("DBG", "parse: buf=%zu headerEnd=%zu reads=%zu readBytes=%zu bodyFNV=%016llx",
           buf.size(), headerEnd, bodyReads, bodyBytes, (unsigned long long)h);
  }

  std::vector<float> outF32(m->outElemsTotal);
  std::vector<uint8_t> int8In;
  if (preQuant) {
    int8In.assign(body.begin(), body.end());
  }
  bool ok = preQuant
               ? infer(*m, int8In.data(), inElems, outF32.data())
               : (quantizeInput(*m, reinterpret_cast<const float*>(body.data()), inElems),
                  infer(*m, m->inBuf.data(), inElems, outF32.data()));
  if (!ok) {
    sendReply(fd, 500, "Internal Server Error", "inference failed");
    return;
  }
  logmsg("DBG", "%s: outF32=%zu", key.c_str(), outF32.size());
  sendReply(fd, 200, "OK", std::string(reinterpret_cast<const char*>(outF32.data()),
                                       outF32.size() * sizeof(float)));
}

static void serve(int listenFd) {
  logmsg("INFO", "listening; models: %zu", g_models.size());
  while (g_running) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listenFd, &rfds);
    struct timeval tv{0, 500000};  // 0.5s — lets the loop observe g_running
    int sel = select(listenFd + 1, &rfds, nullptr, nullptr, &tv);
    if (sel <= 0) continue;
    int fd = accept(listenFd, nullptr, nullptr);
    if (fd < 0) continue;
    handleConnection(fd);
    close(fd);
  }
}

// -------------------------------------------------------------------- main --
static void usage(const char* p) {
  fprintf(stderr,
          "usage: %s --backend <libQnnHtp.so> [options]\n"
          "  --port N                 listen port (default 8089)\n"
          "  --bind ADDR              bind address (default 127.0.0.1)\n"
          "  --clip-context PATH      CLIP ViT-B/32 context binary\n"
          "  --clip-graph NAME        graph name (default clipr37)\n"
          "  --arcface-context PATH   ArcFace w600k_r50 context binary\n"
          "  --arcface-graph NAME     graph name (default arcface37)\n"
          "  --scrfd-context PATH     SCRFD-2.5G face-detector context binary (9 outputs)\n"
          "  --scrfd-graph NAME       graph name (default scrfd)\n",
          p);
}

int main(int argc, char** argv) {
  std::string backend = "libQnnHtp.so";
  std::string bindAddr = "127.0.0.1";
  int port = 8089;
  std::string clipCtx, clipGraph = "clipr37";
  std::string arcCtx, arcGraph = "arcface37";
  std::string scrCtx, scrGraph = "scrfd";
  // NAN = "override not given"; -1 would be ambiguous with a legitimate
  // (negative) offset value.
  double clipInScale = NAN, clipInOff = NAN, clipOutScale = NAN, clipOutOff = NAN;
  double arcInScale = NAN, arcInOff = NAN, arcOutScale = NAN, arcOutOff = NAN;
  uint32_t dataFormat = 1032;

  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* flag) -> const char* {
      if (i + 1 >= argc || std::string(argv[i]) != flag) return nullptr;
      return argv[++i];
    };
    if (const char* v = next("--backend")) backend = v;
    else if (const char* v = next("--port")) port = std::atoi(v);
    else if (const char* v = next("--bind")) bindAddr = v;
    else if (const char* v = next("--clip-context")) clipCtx = v;
    else if (const char* v = next("--clip-graph")) clipGraph = v;
    else if (const char* v = next("--arcface-context")) arcCtx = v;
    else if (const char* v = next("--arcface-graph")) arcGraph = v;
    else if (const char* v = next("--clip-in-scale")) clipInScale = std::atof(v);
    else if (const char* v = next("--clip-in-offset")) clipInOff = std::atof(v);
    else if (const char* v = next("--clip-out-scale")) clipOutScale = std::atof(v);
    else if (const char* v = next("--clip-out-offset")) clipOutOff = std::atof(v);
    else if (const char* v = next("--arcface-in-scale")) arcInScale = std::atof(v);
    else if (const char* v = next("--arcface-in-offset")) arcInOff = std::atof(v);
    else if (const char* v = next("--arcface-out-scale")) arcOutScale = std::atof(v);
    else if (const char* v = next("--arcface-out-offset")) arcOutOff = std::atof(v);
    else if (const char* v = next("--scrfd-context")) scrCtx = v;
    else if (const char* v = next("--scrfd-graph")) scrGraph = v;
    else if (const char* v = next("--data-format")) dataFormat = static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
    else if (std::string(argv[i]) == "--no-device") gNoDevice = true;
    else {
      usage(argv[0]);
      return 2;
    }
  }

  if (clipCtx.empty() && arcCtx.empty() && scrCtx.empty()) {
    usage(argv[0]);
    return 2;
  }

  // The HTP backend finds the DSP skel via cwd — be explicit about it.
  {
    size_t slash = backend.find_last_of('/');
    std::string dir = (slash == std::string::npos) ? "." : backend.substr(0, slash);
    if (dir == "") dir = ".";
    if (chdir(dir.c_str()) != 0) {
      logmsg("WARN", "chdir(%s) failed; relying on inherited cwd", dir.c_str());
    }
  }

  signal(SIGTERM, onSignal);
  signal(SIGINT, onSignal);
  // A half-closed client must not kill the daemon mid-response.
  signal(SIGPIPE, SIG_IGN);

  // ---- load backend interface ----
  void* lib = dlopen(backend.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    logmsg("ERR", "dlopen(%s) failed: %s", backend.c_str(), dlerror());
    return 1;
  }
  typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t***, uint32_t*);
  auto fn = reinterpret_cast<GetProvidersFn>(dlsym(lib, "QnnInterface_getProviders"));
  if (!fn) {
    logmsg("ERR", "dlsym(QnnInterface_getProviders) failed: %s", dlerror());
    return 1;
  }
  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  if (fn(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    logmsg("ERR", "QnnInterface_getProviders failed");
    return 1;
  }
  g_if = *providers[0];

  if (QNNFN(backendCreate)(nullptr, nullptr, &g_backend) != QNN_SUCCESS) {
    logmsg("ERR", "backendCreate failed");
    return 1;
  }
  {
    Qnn_ErrorHandle_t derr = gNoDevice ? QNN_SUCCESS : QNNFN(deviceCreate)(nullptr, nullptr, &g_device);
    if (derr != QNN_SUCCESS) {
      logmsg("WARN", "deviceCreate failed: 0x%x; continuing with NULL device", (unsigned)derr);
      g_device = nullptr;
    }
  }
  logmsg("INFO", "HTP backend initialized (device=%p)", (void*)g_device);

  // ---- tensor specs (ground truth: qairt-dlc-info, QAIRT 2.37 HTP INT8) ----
  // tensor ids from context binary metadata (tools/qnn_sys_introspector)
  ModelSpec clipSpec{"clip", clipGraph.c_str(), "image", "embedding",
                     {1, 3, 224, 224}, {1, 512}, 0.015443762764f, -116, 0.003277973738f, -207,
                     1, 726};
  ModelSpec arcSpec{"arcface", arcGraph.c_str(), "input_1", "_683",
                    {1, 3, 112, 112}, {1, 512}, 0.006889658049f, -136, 0.015603637323f, -123,
                    1, 475};
  // SCRFD-2.5G (buffalo_l detection): 9 outputs in ONNX graph-output order
  // (score/box/kps per FPN level, 2 anchors x 80^2/40^2/20^2). Tensor names,
  // ids, dims, scales, offsets from scrfd_6490_v2.bin metadata
  // (tools/qnn_sys_introspector, QAIRT 2.37.1, uFxp_8, dataFormat 1032).
  static const OutSpec scrfdOuts[] = {
      {"_448", {12800, 1, 1, 1}, 0.0031538952607661486f, 0, 347},
      {"_471", {3200, 1, 1, 1}, 0.0032401620410382748f, 0, 396},
      {"_494", {800, 1, 1, 1}, 0.00014671012468170375f, 0, 445},
      {"_451", {12800, 4, 1, 1}, 0.018793873488903046f, 0, 350},
      {"_474", {3200, 4, 1, 1}, 0.022282257676124573f, 0, 399},
      {"_497", {800, 4, 1, 1}, 0.017743172124028206f, 0, 448},
      {"_454", {12800, 10, 1, 1}, 0.024366198107600212f, -121, 353},
      {"_477", {3200, 10, 1, 1}, 0.030040601268410683f, -126, 402},
      {"_500", {800, 10, 1, 1}, 0.012706567533314228f, -95, 451},
  };
  ModelSpec scrSpec{"scrfd", scrGraph.c_str(), "input_1", "", {1, 3, 640, 640}, {1, 1},
                    0.0078125f, -128, 0.0f, 0, 1, 0, scrfdOuts, 9};
  if (std::isfinite(clipInScale) && clipInScale > 0) clipSpec.inScale = static_cast<float>(clipInScale);
  if (std::isfinite(clipInOff)) clipSpec.inOffset = static_cast<int32_t>(clipInOff);
  if (std::isfinite(clipOutScale) && clipOutScale > 0) clipSpec.outScale = static_cast<float>(clipOutScale);
  if (std::isfinite(clipOutOff)) clipSpec.outOffset = static_cast<int32_t>(clipOutOff);
  if (std::isfinite(arcInScale) && arcInScale > 0) arcSpec.inScale = static_cast<float>(arcInScale);
  if (std::isfinite(arcInOff)) arcSpec.inOffset = static_cast<int32_t>(arcInOff);
  if (std::isfinite(arcOutScale) && arcOutScale > 0) arcSpec.outScale = static_cast<float>(arcOutScale);
  if (std::isfinite(arcOutOff)) arcSpec.outOffset = static_cast<int32_t>(arcOutOff);
  gTensorDataFormat = dataFormat;

  // Load order = VTCM priority: CLIP + ArcFace (existing production models)
  // first, SCRFD last — if the Hexagon VTCM cannot host all three contexts,
  // the newest one (detection) fails to load and degrades to CPU via the
  // QnnSession fallback while the other two keep serving.
  if (!clipCtx.empty()) loadModel(&clipSpec, clipCtx);
  if (!arcCtx.empty()) loadModel(&arcSpec, arcCtx);
  if (!scrCtx.empty()) loadModel(&scrSpec, scrCtx);
  if (g_models.size() == 0) {
    logmsg("ERR", "no models loaded — exiting");
    return 1;
  }

  // ---- listen ----
  int listenFd = socket(AF_INET, SOCK_STREAM, 0);
  if (listenFd < 0) {
    logmsg("ERR", "socket failed");
    return 1;
  }
  int one = 1;
  setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, bindAddr.c_str(), &addr.sin_addr) != 1) {
    logmsg("ERR", "invalid --bind address: %s", bindAddr.c_str());
    return 1;
  }
  if (bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(listenFd, 16) != 0) {
    logmsg("ERR", "bind/listen on %s:%d failed", bindAddr.c_str(), port);
    return 1;
  }

  serve(listenFd);

  logmsg("INFO", "shutting down");
  close(listenFd);
  // Tear down in dependency order: contexts (newest first), then device,
  // then backend. Abrupt process death without this sequence is the
  // documented suspect for CDSP wedges that require a power cycle.
  for (auto it = g_models.rbegin(); it != g_models.rend(); ++it) {
    if (it->ctx && QNNFN(contextFree)) {
      Qnn_ErrorHandle_t e = QNNFN(contextFree)(it->ctx, nullptr);
      logmsg("INFO", "contextFree(%s) = 0x%x", it->spec->key, (unsigned)e);
    }
    it->ctx = nullptr;
  }
  if (g_device && QNNFN(deviceFree)) {
    Qnn_ErrorHandle_t e = QNNFN(deviceFree)(g_device);
    logmsg("INFO", "deviceFree = 0x%x", (unsigned)e);
    g_device = nullptr;
  }
  if (g_backend && QNNFN(backendFree)) {
    Qnn_ErrorHandle_t e = QNNFN(backendFree)(g_backend);
    logmsg("INFO", "backendFree = 0x%x", (unsigned)e);
    g_backend = nullptr;
  }
  return 0;
}
