// qnn_meta_test — run one inference using tensor descriptors taken directly
// from the context binary's QnnSystem metadata (the canonical SampleApp flow).
//
// usage: qnn_meta_test <libQnnSystem.so> <libQnnHtp.so> <context.bin> <graph> <in.raw(int8|float32)> <out.raw>
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

  // ---------- QnnSystem metadata ----------
  void* slib = dlopen(sysLib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!slib) die(std::string("dlopen libQnnSystem: ") + dlerror());
  typedef Qnn_ErrorHandle_t (*SysGetProviders)(const QnnSystemInterface_t***, uint32_t*);
  auto sysGet = reinterpret_cast<SysGetProviders>(dlsym(slib, "QnnSystemInterface_getProviders"));
  if (!sysGet) die("dlsym QnnSystemInterface_getProviders");
  const QnnSystemInterface_t** sprov = nullptr;
  uint32_t sn = 0;
  if (sysGet(&sprov, &sn) != QNN_SUCCESS || sn == 0) die("sysGetProviders");
  QnnSystemInterface_t siface = *sprov[0];
  auto& S = siface.QNN_SYSTEM_INTERFACE_VER_NAME;

  QnnSystemContext_Handle_t sctx = nullptr;
  if (S.systemContextCreate(&sctx) != QNN_SUCCESS) die("systemContextCreate");
  const QnnSystemContext_BinaryInfo_t* bi = nullptr;
  Qnn_ContextBinarySize_t biSz = 0;
  if (S.systemContextGetBinaryInfo(sctx, buf.data(), static_cast<uint64_t>(buf.size()), &bi,
                                   &biSz) != QNN_SUCCESS)
    die("systemContextGetBinaryInfo");
  if (bi->version != QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) die("binary info v3 expected");
  auto* v3 = &bi->contextBinaryInfoV3;
  Qnn_Tensor_t* inT = nullptr, *outT = nullptr;
  for (uint32_t g = 0; g < v3->numGraphs; ++g) {
    auto* gi = &v3->graphs[g];
    if (gi->version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) die("graph info v3 expected");
    if (std::string(gi->graphInfoV3.graphName) != graphName) continue;
    if (gi->graphInfoV3.numGraphInputs != 1 || gi->graphInfoV3.numGraphOutputs != 1)
      die("expected 1 input / 1 output");
    inT = &gi->graphInfoV3.graphInputs[0];
    outT = &gi->graphInfoV3.graphOutputs[0];
  }
  if (!inT || !outT) die("graph not found in metadata");

  // ---------- deep-copy descriptors (SampleApp pattern) ----------
  auto copyTensor = [](Qnn_Tensor_t& dst, const Qnn_Tensor_t& src) {
    dst = src;  // full copy: version, name, type, dataFormat, dataType, quantizeParams,
                // rank, dims, memType, id...
  };
  Qnn_Tensor_t inTensor, outTensor;
  copyTensor(inTensor, *inT);
  copyTensor(outTensor, *outT);

  // dims + sizes
  auto dimsOf = [](const Qnn_Tensor_t& t) {
    std::vector<uint32_t> d;
    if (t.version == QNN_TENSOR_VERSION_1) {
      for (uint32_t i = 0; i < t.v1.rank; ++i) d.push_back(t.v1.dimensions[i]);
    } else {
      for (uint32_t i = 0; i < t.v2.rank; ++i) d.push_back(t.v2.dimensions[i]);
    }
    return d;
  };
  auto encOf = [](const Qnn_Tensor_t& t) {
    if (t.version == QNN_TENSOR_VERSION_1) return t.v1.quantizeParams;
    return t.v2.quantizeParams;
  };
  std::vector<uint32_t> inDims = dimsOf(inTensor);
  std::vector<uint32_t> outDims = dimsOf(outTensor);
  size_t inElems = 1, outElems = 1;
  for (uint32_t d : inDims) inElems *= d;
  for (uint32_t d : outDims) outElems *= d;
  Qnn_QuantizeParams_t inEnc = encOf(inTensor), outEnc = encOf(outTensor);
  float inScale = inEnc.scaleOffsetEncoding.scale;
  int32_t inOff = inEnc.scaleOffsetEncoding.offset;
  float outScale = outEnc.scaleOffsetEncoding.scale;
  int32_t outOff = outEnc.scaleOffsetEncoding.offset;
  fprintf(stderr, "meta: in '%s' dims=[%zu elems] scale=%.17g off=%d | out '%s' [%zu] scale=%.17g off=%d\n",
          inTensor.v1.name, inElems, (double)inScale, inOff, outTensor.v1.name, outElems,
          (double)outScale, outOff);

  // ---------- read float input, quantize ----------
  FILE* fin = fopen(inPath.c_str(), "rb");
  if (!fin) die("open input");
  size_t inBytes = inElems * sizeof(float);
  std::vector<float> inF32(inElems);
  if (fread(inF32.data(), 1, inBytes, fin) != inBytes) die("read input");
  fclose(fin);
  std::vector<uint8_t> inInt8(inElems), outInt8(outElems);
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
