// qnn_hand_test — run one inference using tensor descriptors taken directly
// from the context binary's QnnSystem metadata (the canonical SampleApp flow).
//
// usage: qnn_hand_test <libQnnSystem.so> <libQnnHtp.so> <context.bin> <graph> <in.raw(int8|float32)> <out.raw>
//
// The input file is float32; it is quantized with the input encoding from the
// metadata (round-half-away, double precision) and the int8 output is
// dequantized to float32 with the output encoding.

#include <QNN/QnnInterface.h>
#include <QNN/System/QnnSystemCommon.h>
#include <QNN/System/QnnSystemContext.h>
#include <QNN/System/QnnSystemInterface.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>

static void die(const std::string& m) {
  fprintf(stderr, "FATAL: %s\n", m.c_str());
  exit(1);
}

int main(int argc, char** argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s <libQnnSystem.so> <libQnnHtp.so> <context.bin> <graph> <in.raw> <out.raw>\n", argv[0]);
    return 2;
  }
  std::string sysLib = argv[1], backendLib = argv[2], ctxPath = argv[3], graphName = argv[4];
  std::string inPath = argv[5], outPath = argv[6];

  // ---------- backend ----------
  void* blib = dlopen(backendLib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!blib) die(std::string("dlopen backend: ") + dlerror());
  typedef Qnn_ErrorHandle_t (*QnnGetProviders)(const QnnInterface_t***, uint32_t*);
  auto getProviders = reinterpret_cast<QnnGetProviders>(dlsym(blib, "QnnInterface_getProviders"));
  if (!getProviders) die("dlsym QnnInterface_getProviders");
  const QnnInterface_t** providers = nullptr;
  uint32_t n = 0;
  if (getProviders(&providers, &n) != QNN_SUCCESS || n == 0) die("getProviders");
  QnnInterface_t iface = *providers[0];
  auto& I = iface.v2_28;

  Qnn_BackendHandle_t backend = nullptr;
  if (I.backendCreate(nullptr, nullptr, &backend) != QNN_SUCCESS) die("backendCreate");

  // ---------- context from file buffer ----------
  FILE* f = fopen(ctxPath.c_str(), "rb");
  if (!f) die("open context");
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> buf(static_cast<size_t>(sz));
  if (fread(buf.data(), 1, buf.size(), f) != buf.size()) die("read context");
  fclose(f);

  Qnn_ContextHandle_t ctx = nullptr;
  if (I.contextCreateFromBinary(backend, nullptr, nullptr, buf.data(), buf.size(), &ctx,
                                nullptr) != QNN_SUCCESS)
    die("contextCreateFromBinary");
  Qnn_GraphHandle_t graph = nullptr;
  if (I.graphRetrieve(ctx, graphName.c_str(), &graph) != QNN_SUCCESS) die("graphRetrieve");

  // ---------- hand-rolled tensors (daemon-style) ----------
  static uint32_t inDimsA[4] = {1, 3, 224, 224};
  static uint32_t outDimsA[2] = {1, 512};
  const char* inNameA = "image";
  const char* outNameA = "embedding";
  float inScale = 0.015443762764f;
  int32_t inOff = -116;
  float outScale = 0.003277973738f;
  int32_t outOff = -207;
  size_t inElems = 1 * 3 * 224 * 224;
  size_t outElems = 512;
  auto mkT = [](const char* nm, uint32_t type, const uint32_t* dims, uint32_t rank,
                float sc, int32_t off, uint8_t* data, uint32_t bytes, uint32_t id) {
    Qnn_Tensor_t t{};
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.id = id;
    t.v1.name = nm;
    t.v1.type = static_cast<Qnn_TensorType_t>(type);
    t.v1.dataFormat = static_cast<Qnn_TensorDataFormat_t>(getenv("DF") ? atoi(getenv("DF")) : 1032);
    t.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
    t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    t.v1.quantizeParams.scaleOffsetEncoding.scale = sc;
    t.v1.quantizeParams.scaleOffsetEncoding.offset = off;
    t.v1.rank = rank;
    t.v1.dimensions = const_cast<uint32_t*>(dims);
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = data;
    t.v1.clientBuf.dataSize = bytes;
    return t;
  };
  std::vector<uint8_t> inInt8(inElems), outInt8(outElems);
  Qnn_Tensor_t inTensor = mkT(inNameA, QNN_TENSOR_TYPE_APP_WRITE, inDimsA, 4, inScale, inOff,
                              inInt8.data(), (uint32_t)inElems, 1);
  Qnn_Tensor_t outTensor = mkT(outNameA, QNN_TENSOR_TYPE_APP_READ, outDimsA, 2, outScale, outOff,
                               outInt8.data(), (uint32_t)outElems, 726);
  fprintf(stderr, "hand tensors built (df=%d)\n", (int)inTensor.v1.dataFormat);

  // ---------- read float input, quantize ----------
  FILE* fin = fopen(inPath.c_str(), "rb");
  if (!fin) die("open input");
  size_t inBytes = inElems * sizeof(float);
  std::vector<float> inF32(inElems);
  if (fread(inF32.data(), 1, inBytes, fin) != inBytes) die("read input");
  fclose(fin);
  for (size_t i = 0; i < inElems; ++i) {
    double q = std::round(static_cast<double>(inF32[i]) / inScale - inOff);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    inInt8[i] = static_cast<uint8_t>(q);
  }

  // ---------- field overrides (bisecting which field the backend cares about) ----------
  auto env32 = [](const char* k) -> long { const char* v = getenv(k); return v ? strtol(v, nullptr, 0) : 0; };
  if (env32("OV_QUANT")) {
    // hand-rolled quantizeParams like the daemon: same values, different float bits
    inTensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    inTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    inTensor.v1.quantizeParams.scaleOffsetEncoding.scale = 0.015443762764f;
    inTensor.v1.quantizeParams.scaleOffsetEncoding.offset = -116;
    outTensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    outTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    outTensor.v1.quantizeParams.scaleOffsetEncoding.scale = 0.003277973738f;
    outTensor.v1.quantizeParams.scaleOffsetEncoding.offset = -207;
    fprintf(stderr, "OV: quantizeParams overridden with daemon literals\n");
  }
  if (long df = env32("OV_DF")) {
    inTensor.v1.dataFormat = static_cast<Qnn_TensorDataFormat_t>(df);
    outTensor.v1.dataFormat = static_cast<Qnn_TensorDataFormat_t>(df);
    fprintf(stderr, "OV: dataFormat=%ld\n", df);
  }
  if (long id = env32("OV_ID")) {
    inTensor.v1.id = static_cast<uint32_t>(id);
    outTensor.v1.id = 0;
    fprintf(stderr, "OV: in.id=%ld\n", id);
  }
  if (long nm = env32("OV_NAME")) {
    inTensor.v1.name = "image";  // same as metadata for clip; no-op sanity
    fprintf(stderr, "OV: name set\n");
  }

  {
    const char* di = getenv("DUMP_INT8");
    if (di) {
      FILE* di_f = fopen(di, "wb");
      fwrite(inInt8.data(), 1, inInt8.size(), di_f);
      fclose(di_f);
      fprintf(stderr, "dumped %zu int8 bytes\n", inInt8.size());
    }
  }

  // ---------- attach buffers ----------
  inTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
  inTensor.v1.clientBuf.data = inInt8.data();
  inTensor.v1.clientBuf.dataSize = static_cast<uint32_t>(inElems);
  outTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
  outTensor.v1.clientBuf.data = outInt8.data();
  outTensor.v1.clientBuf.dataSize = static_cast<uint32_t>(outElems);

  {
    const char* dp = getenv("DUMP_TENSORS");
    if (dp) {
      FILE* df = fopen(dp, "wb");
      fwrite(&inTensor, 1, sizeof(inTensor), df);
      fwrite(&outTensor, 1, sizeof(outTensor), df);
      fclose(df);
      fprintf(stderr, "dumped tensors (%zu bytes each)\n", sizeof(inTensor));
    }
  }

  // ---------- execute ----------
  Qnn_ErrorHandle_t err = I.graphExecute(graph, &inTensor, 1, &outTensor, 1, nullptr, nullptr);
  if (err != QNN_SUCCESS) {
    fprintf(stderr, "graphExecute failed: 0x%x\n", (unsigned)err);
    return 1;
  }

  // ---------- dequantize + write ----------
  FILE* fout = fopen(outPath.c_str(), "wb");
  if (!fout) die("open output");
  std::vector<float> outF32(outElems);
  for (size_t i = 0; i < outElems; ++i) {
    outF32[i] = (static_cast<float>(static_cast<int32_t>(outInt8[i])) + static_cast<float>(outOff)) *
                outScale;
  }
  fwrite(outF32.data(), 1, outF32.size() * sizeof(float), fout);
  fclose(fout);
  fprintf(stderr, "wrote %zu floats to %s\n", outElems, outPath.c_str());
  return 0;
}
