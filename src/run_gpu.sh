#!/usr/bin/env bash
# Run a python script in the WSL GPU venv with the CUDA libraries on the path.
# Usage (from WSL):  ./run_gpu.sh train_dual_protocol.py
# The LD_LIBRARY_PATH fix is required because TF needs the WSL driver stub
# (/usr/lib/wsl/lib/libcuda.so.1) plus the pip-installed nvidia/*/lib dirs.
set -e
source ~/ecg-gpu/bin/activate
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
NVLIBS=$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVLIBS}${LD_LIBRARY_PATH}"
export TF_CPP_MIN_LOG_LEVEL=1
python -u "$@"
