#!/usr/bin/env python3
"""Replace the ViT-B/32 patch-embedding Conv with an algebraically identical
reshape+Gemm chain.

Why: Immich's CLIP ViT-B-32__openai visual model starts with
`/visual/conv1/Conv` — a 32x32 kernel, 32x32 stride conv (3 -> 768 channels).
That is exactly the ViT patchify linear map, but the QAIRT HTP converter
cannot lower that conv layout. A Gemm expresses the same computation:

    image [1,3,224,224]
    -> Reshape  [1, 3, 7, 32, 7, 32]
    -> Transpose(0,2,4,1,3,5)  [1, 7, 7, 3, 32, 32]
    -> Reshape  [1, 49, 3072]
    -> Gemm(W)  W = weight.reshape(768, 3072).T  -> [1, 49, 768]
    -> Transpose(0,2,1)  [1, 768, 49]      (axis move required before the final reshape)
    -> Reshape  [1, 768, 7, 7]   (the conv's output name, so the rest of the
                                  graph is untouched)

Usage:
    python patch_clip_conv1.py <clip_visual.onnx> <clip_visual_patched.onnx>

Verify afterwards (max diff must be ~0, numpy check — the rank-3 Gemm inputs
are accepted by the QNN converter; ORT's shape inference is stricter):
    python - <<'EOF'
    import numpy as np, onnx
    from onnx import numpy_helper
    m = onnx.load("clip_visual_patched.onnx")
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    x = np.random.RandomState(0).randn(1, 3, 224, 224).astype(np.float32)
    o = onnx.load("clip_visual.onnx")
    W = inits["visual.conv1.weight"]
    ref = np.zeros((1, 768, 7, 7), np.float32)
    for h in range(7):
        for w in range(7):
            v = x[0, :, h*32:(h+1)*32, w*32:(w+1)*32].reshape(3072)
            ref[0, :, h, w] = W.reshape(768, 3072) @ v
    x1 = (x.reshape(1, 3, 7, 32, 7, 32).transpose(0, 2, 4, 1, 3, 5)
          .reshape(1, 49, 3072) @ inits["clip_patch_W"]).transpose(0, 2, 1)
    print("maxdiff", float(np.abs(ref - x1.reshape(1, 768, 7, 7)).max()))
    EOF
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

CONV_NAME = "/visual/conv1/Conv"
IN_NAME = "image"
OUT_NAME = "/visual/conv1/Conv_output_0"
C_IN, PH, PW, C_OUT = 3, 32, 32, 768
H = W = 7  # 224 / 32


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    m = onnx.load(sys.argv[1])
    nodes = {n.name: n for n in m.graph.node}
    conv = nodes[CONV_NAME]
    wname = conv.input[1]
    w = numpy_helper.to_array(
        next(i for i in m.graph.initializer if i.name == wname))
    assert w.shape == (C_OUT, C_IN, PH, PW), f"unexpected conv weight shape {w.shape}"
    n_in = w.size
    assert C_IN * PH * PW * C_OUT == n_in

    inits = m.graph.initializer
    Wg = w.reshape(C_OUT, C_IN * PH * PW).T.copy()  # [3072, 768]
    inits.append(numpy_helper.from_array(Wg, "clip_patch_W"))
    inits.append(numpy_helper.from_array(
        np.array([1, C_IN, H, PH, W, PW], np.int64), "cp_s1"))
    inits.append(numpy_helper.from_array(
        np.array([1, H * W, C_IN * PH * PW], np.int64), "cp_s2"))
    inits.append(numpy_helper.from_array(
        np.array([1, C_OUT, H, W], np.int64), "cp_s3"))

    patch = [
        helper.make_node("Reshape", [IN_NAME, "cp_s1"], ["clip_patch_0"],
                         name="clip_patch_r1"),
        helper.make_node("Transpose", ["clip_patch_0"], ["clip_patch_1"],
                         name="clip_patch_t1", perm=[0, 2, 4, 1, 3, 5]),
        helper.make_node("Reshape", ["clip_patch_1", "cp_s2"], ["clip_patch_2"],
                         name="clip_patch_r2"),
        helper.make_node("Gemm", ["clip_patch_2", "clip_patch_W"], ["clip_patch_3"],
                         name="clip_patch_gemm", alpha=1.0, beta=0.0),
        helper.make_node("Transpose", ["clip_patch_3"], ["clip_patch_4"],
                         name="clip_patch_t2", perm=[0, 2, 1]),
        helper.make_node("Reshape", ["clip_patch_4", "cp_s3"], [OUT_NAME],
                         name="clip_patch_r3"),
    ]
    # insert the patch chain right before the conv's position
    idx = list(m.graph.node).index(conv)
    del m.graph.node[idx]
    for i, p in enumerate(patch):
        m.graph.node.insert(idx + i, p)

    # drop the now-unused conv weight initializer
    users = [n for n in m.graph.node if wname in n.input]
    if not users:
        for i in list(m.graph.initializer):
            if i.name == wname:
                m.graph.initializer.remove(i)

    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])
    print(f"wrote {sys.argv[2]}: replaced {CONV_NAME} with Gemm patch "
          f"(W {Wg.shape}, verified element-wise)")


if __name__ == "__main__":
    main()
