#!/usr/bin/env python3
"""Compile a (quantized) QAIRT DLC into an HTP context binary for the
Radxa Dragon Q6A (QCS6490 / SM7325, HTP v68).

Uses the QAIRT Python API (qairt.compile). The CLI generator's --config_file
does NOT apply device/SoC settings (byte-identical output), so the Python
API's CompileConfig is the supported way to pin the SoC target:

    soc_details="chipset:SM7325;dsp_arch:v68;soc_model:35"

  * QNN_SOC_MODEL_SM7325 = 35 (see include/QNN/QnnTypes.h in the SDK docs)
  * dsp_arch values: v66 v68 v69 v73 v75 v79 v81
  * HTP is INT8-only on this SoC: the DLC must come from qairt-quantizer
    (--act_bitwidth 8 --weights_bitwidth 8 --use_per_channel_quantization).

The resulting .bin is loaded with QnnContext_createFromBinary by the
qnn-dsp-daemon (or qnn-net-run for ground-truth references).

Usage:
    python compile_htp.py <model.dlc> <graph_name> [vtcm_mb] [out_dir]

Defaults: vtcm_mb=2 (8 MB VTCM on QCS6490, 2 MB per graph is the working
setting used for the production CLIP + ArcFace binaries), out_dir=<cwd>.

Environment:
    QNN_SDK_ROOT    SDK root (required for the save step)
    LD_LIBRARY_PATH must include <SDK>/lib/x86_64-linux-clang and the
                    venv python's lib dir (libpython3.10.so.1.0)

The compiled file is written as <out_dir>/<dlc_stem>.bin (the API uses the
DLC's own module name) and renamed to <out_dir>/<graph_name>.bin by this
script so each model has a stable, explicit name.
"""
import glob
import os
import shutil
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    dlc, graph = sys.argv[1], sys.argv[2]
    vtcm = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 2
    out_dir = sys.argv[4] if len(sys.argv) > 4 else os.getcwd()

    # The qairt package's argparser-based entrypoints sniff sys.argv; the
    # API path we use here does not want it.
    sys.argv = ["qairt-compile"]
    import qairt
    from qairt.api.compiler.backends.htp import HtpGraphConfig
    from qairt.api.compiler.config import CompileConfig

    model = qairt.load(dlc)
    compiled = qairt.compile(
        model,
        config=CompileConfig(
            backend="HTP",
            soc_details="chipset:SM7325;dsp_arch:v68;soc_model:35",
            graph_custom_configs=[
                HtpGraphConfig(name=graph, vtcm_size_in_mb=vtcm)
            ],
        ),
    )
    compiled.save(out_dir)
    stem = os.path.splitext(os.path.basename(dlc))[0]
    produced = os.path.join(out_dir, stem + ".bin")
    final = os.path.join(out_dir, graph + ".bin")
    shutil.move(produced, final)
    print(f"WROTE {final} ({os.path.getsize(final)} bytes)")


if __name__ == "__main__":
    main()
