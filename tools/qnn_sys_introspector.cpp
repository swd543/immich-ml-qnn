// qnn_sys_introspector — dump a QNN context binary's graph I/O tensor
// descriptors using libQnnSystem.so (no DSP required).
//
// usage: qnn_sys_introspector <libQnnSystem.so> <context.bin>
//
// Build: g++ -O2 -std=c++17 -I<include> qnn_sys_introspector.cpp -o qnn_sys_introspector -ldl

#include <QNN/QnnInterface.h>
#include <QNN/System/QnnSystemCommon.h>
#include <QNN/System/QnnSystemContext.h>
#include <QNN/System/QnnSystemInterface.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <dlfcn.h>

static const char* dataTypeName(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_32: return "fPxp_32";
    case QNN_DATATYPE_FLOAT_16: return "fPxp_16";
    case QNN_DATATYPE_SFIXED_POINT_32: return "sFxp_32";
    case QNN_DATATYPE_UFIXED_POINT_8: return "uFxp_8";
    case QNN_DATATYPE_SFIXED_POINT_8: return "sFxp_8";
    default: return "?";
  }
}

static void printTensor(const char* kind, const Qnn_Tensor_t* t) {
  // Handle v1 (and v2, which shares the leading fields) defensively.
  uint32_t version = t->version;
  const char* name = nullptr;
  uint32_t rank = 0;
  uint32_t* dims = nullptr;
  Qnn_DataType_t dt = QNN_DATATYPE_UNDEFINED;
  Qnn_TensorType_t type = QNN_TENSOR_TYPE_UNDEFINED;
  Qnn_TensorMemType_t memType = QNN_TENSORMEMTYPE_UNDEFINED;
  Qnn_QuantizeParams_t qp{};
  if (version == QNN_TENSOR_VERSION_1) {
    name = t->v1.name;
    rank = t->v1.rank;
    dims = t->v1.dimensions;
    dt = t->v1.dataType;
    type = t->v1.type;
    memType = t->v1.memType;
    qp = t->v1.quantizeParams;
  } else {
    name = t->v2.name;
    rank = t->v2.rank;
    dims = t->v2.dimensions;
    dt = t->v2.dataType;
    type = t->v2.type;
    memType = t->v2.memType;
    qp = t->v2.quantizeParams;
  }
  char dimsBuf[256] = "";
  char* p = dimsBuf;
  *p = 0;
  for (uint32_t i = 0; i < rank; ++i) {
    p += snprintf(p, 256 - (p - dimsBuf), "%s%u", i ? "x" : "", dims ? dims[i] : 0);
  }
  printf("    %-10s %-16s %-8s type=%d dataFormat=%u rank=%u dims=[%s] memType=%u version=%u\n",
         kind, name ? name : "?", dataTypeName(dt), (int)type, (unsigned)dt, (unsigned)rank,
         dimsBuf, (unsigned)memType, (unsigned)version);
  if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    printf("             quantize: encodingDef=%u enc=%u scale=%.17g offset=%d\n",
           (unsigned)qp.encodingDefinition, (unsigned)qp.quantizationEncoding,
           (double)qp.scaleOffsetEncoding.scale, (int)qp.scaleOffsetEncoding.offset);
  } else {
    printf("             quantize: encodingDef=%u enc=%u (non-scale-offset)\n",
           (unsigned)qp.encodingDefinition, (unsigned)qp.quantizationEncoding);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <libQnnSystem.so> <context.bin>\n", argv[0]);
    return 2;
  }
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    fprintf(stderr, "dlopen(%s): %s\n", argv[1], dlerror());
    return 1;
  }
  typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnSystemInterface_t***, uint32_t*);
  auto getProviders = reinterpret_cast<GetProvidersFn>(dlsym(lib, "QnnSystemInterface_getProviders"));
  if (!getProviders) {
    fprintf(stderr, "dlsym(QnnSystemInterface_getProviders): %s\n", dlerror());
    return 1;
  }
  const QnnSystemInterface_t** providers = nullptr;
  uint32_t n = 0;
  if (getProviders(&providers, &n) != QNN_SUCCESS || n == 0) {
    fprintf(stderr, "getProviders failed\n");
    return 1;
  }
  QnnSystemInterface_t wrap = *providers[0];
  auto& I = wrap.QNN_SYSTEM_INTERFACE_VER_NAME;

  QnnSystemContext_Handle_t ctx = nullptr;
  if (I.systemContextCreate(&ctx) != QNN_SUCCESS) {
    fprintf(stderr, "systemContextCreate failed\n");
    return 1;
  }

  FILE* f = fopen(argv[2], "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> buf(static_cast<size_t>(sz));
  if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    fprintf(stderr, "short read\n");
    return 1;
  }
  fclose(f);

  const QnnSystemContext_BinaryInfo_t* bi = nullptr;
  Qnn_ContextBinarySize_t biSize = 0;
  if (I.systemContextGetBinaryInfo(ctx, buf.data(), static_cast<uint64_t>(buf.size()), &bi,
                                   &biSize) != QNN_SUCCESS || !bi) {
    fprintf(stderr, "systemContextGetBinaryInfo failed\n");
    return 1;
  }

  printf("context: %s\n", argv[2]);
  if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_V3) {
    auto* v3 = &bi->contextBinaryInfoV3;
    printf("  backendId=%u buildId=%s socVersion=%s blobSize=%lu socModel=%u\n", v3->backendId,
           v3->buildId ? v3->buildId : "?", v3->socVersion ? v3->socVersion : "?",
           (unsigned long)v3->contextBlobSize, v3->socModel);
    printf("  graphs=%u\n", v3->numGraphs);
    for (uint32_t g = 0; g < v3->numGraphs; ++g) {
      auto* gi = &v3->graphs[g];
      const char* gname = nullptr;
      uint32_t nIn = 0, nOut = 0;
      Qnn_Tensor_t* ins = nullptr, *outs = nullptr;
      if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        gname = gi->graphInfoV1.graphName;
        nIn = gi->graphInfoV1.numGraphInputs;
        ins = gi->graphInfoV1.graphInputs;
        nOut = gi->graphInfoV1.numGraphOutputs;
        outs = gi->graphInfoV1.graphOutputs;
      } else if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        gname = gi->graphInfoV2.graphName;
        nIn = gi->graphInfoV2.numGraphInputs;
        ins = gi->graphInfoV2.graphInputs;
        nOut = gi->graphInfoV2.numGraphOutputs;
        outs = gi->graphInfoV2.graphOutputs;
      } else if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        gname = gi->graphInfoV3.graphName;
        nIn = gi->graphInfoV3.numGraphInputs;
        ins = gi->graphInfoV3.graphInputs;
        nOut = gi->graphInfoV3.numGraphOutputs;
        outs = gi->graphInfoV3.graphOutputs;
      }
      printf("  graph[%u] '%s' (graphinfo v%u): inputs=%u outputs=%u\n", g, gname ? gname : "?",
             (unsigned)gi->version, nIn, nOut);
      for (uint32_t i = 0; i < nIn; ++i) printTensor("IN", &ins[i]);
      for (uint32_t o = 0; o < nOut; ++o) printTensor("OUT", &outs[o]);
    }
  } else {
    fprintf(stderr, "unexpected binary info version %u\n", (unsigned)bi->version);
    return 1;
  }

  I.systemContextFree(ctx);
  return 0;
}
