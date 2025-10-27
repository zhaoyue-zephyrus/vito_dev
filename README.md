# Vito

## Installation

```bash
# install ffmpeg with nvenc
conda create --name vito python=3.12 -y
conda install libnpp cuda-nvrtc -c nvidia/label/cuda-12.8.0
conda install ffmpeg==7.1.1 -c conda-forge

uv venv
MAX_JOB=4 uv sync  # it'll take a while (mostly for `flash_attn`)
# install torchcodec with nvenc
export LD_LIBRARY_PATH=$CONDA_HOME/envs/vito/lib/
uv pip install torchcodec==0.8.0+cu128 --index-url=https://download.pytorch.org/whl/cu128
```

