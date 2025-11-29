# DWPose video
_simple implementation for video to animated DWPose_  
![pose](https://github.com/legraphista/dwpose-video/assets/962643/7161a2ba-c19c-4c3e-94fc-ef584a60bdf7)

## Install

```bash
# 1. 创建环境
conda create -n dwpose python=3.10 -y
conda activate dwpose

# 2. 安装 onnxruntime-gpu
pip install -r requirements.txt

# 3. 安装 NVIDIA 官方提供的 CUDA 12 运行时核心库和 cuDNN
# 这些包会包含你缺失的 libcublasLt.so.12 和 libcudnn*.so
# pip 会自动拉取适配当前架构的最新版本 (即适配 CUDA 12.x)
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12

# 4. 配置环境变量
# ONNX Runtime 默认不会去 Python 的 site-packages 目录寻找这些库，必须手动链接
# 请将以下命令根据你的 shell 类型 (bash/zsh) 运行，或者写入 ~/.bashrc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# 5. 验证安装
python -c "import onnxruntime as ort; print('Device:', ort.get_device()); print('Providers:', ort.get_available_providers())"
```

## Models
You should be able to find the models [here](https://github.com/IDEA-Research/DWPose/tree/onnx#-dwpose-for-controlnet) in the controlnet section

## Usage

```bash
python main.py INPUT.mp4 POSE.mp4
```
Output is h264 RGB variant (is not be playable by all video players, preview with VLC or ffplay)


