import os
import onnxruntime as ort

# 1. 模型路径（根据你当前工作目录调整）
model_path = "models/dw-ll_ucoco_384.onnx"

print("模型路径:", os.path.abspath(model_path))
print("文件是否存在:", os.path.exists(model_path))

# 2. 打印 onnxruntime 版本
print("onnxruntime version:", ort.__version__)

# 3. 尝试优先使用 CUDAExecutionProvider，其次 CPUExecutionProvider
sess = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# 4. 实际启用的 Provider
providers = sess.get_providers()
print("providers =", providers)

if "CUDAExecutionProvider" in providers:
    print("✅ 实际正在使用 GPU (CUDAExecutionProvider)")
else:
    print("⚠ 当前只使用 CPUExecutionProvider，没有启用 GPU")
