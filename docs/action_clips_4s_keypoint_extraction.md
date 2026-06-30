# action_clips_4s keypoint 提取方案

## 目标

把 `/home/swq/Code/ePBSI/data/action_clips_4s/` 下的 4 秒视频片段转成 DWPose 关键点序列，并保持原有 split/center/class/session 目录结构：

- 输入：`action_clips_4s/{train,val,test}/.../*.mp4`
- 输出：`action_clips_4s_keypoint/{train,val,test}/.../*.npz`

每个视频固定均匀采样 32 帧，输出一个 `.npz` 文件。

## 当前数据规模

当前本机已确认视频数量：

- `train/shanghai`: 48425
- `val/shanghai`: 11751
- `test/linfen`: 9457
- 合计：69633

## 模型与环境

模型文件位于：

```bash
/home/swq/models/dw-ll_ucoco_384.onnx
/home/swq/models/yolox_l.onnx
```

正式运行使用 `dwpose` conda 环境。当前 base 环境能看到 `CUDAExecutionProvider`，但缺少 CUDA 12/cuDNN 动态库路径，会回退到 CPU；`dwpose` 环境已验证可以实际启用 GPU。

GPU provider 验证命令：

```bash
cd /home/swq
conda run -n dwpose python /home/swq/Code/dwpose-video/test_onnxruntime.py
```

预期输出包含：

```text
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## 输出格式

每个输出 `.npz` 包含：

- `keypoints`: `float32`，shape 为 `(32, 134, 3)`
- `frame_indices`: `int32`，本视频实际采样的原始帧序号
- `metadata`: JSON 字符串

`keypoints[..., 0]` 是原始视频坐标系的 `x`，`keypoints[..., 1]` 是 `y`，`keypoints[..., 2]` 是置信度。DWPose 后处理会插入 neck，因此关键点数是 134。无人或缺失关键点位置保留 0，置信度为 0。

每帧若检测到多个人，脚本选择该帧关键点置信度总和最高的人作为主序列。

## smoke 验证

先处理 `test/linfen` 排序后的前 3 个视频：

```bash
cd /home/swq/Code/dwpose-video
conda run -n dwpose python batch_process_videos_ultra.py \
  --input /home/swq/Code/ePBSI/data/action_clips_4s/test/linfen \
  --output /home/swq/Code/ePBSI/data/action_clips_4s_keypoint/test/linfen \
  --model-dir /home/swq/models \
  --device cuda \
  --frames-per-clip 32 \
  --batch-size 8 \
  --no-video \
  --format npz \
  --direct-keypoint-output \
  --limit 3
```

检查 smoke 输出：

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/home/swq/Code/ePBSI/data/action_clips_4s_keypoint/test/linfen')
files = sorted(root.rglob('*.npz'))[:3]
for path in files:
    data = np.load(path)
    metadata = json.loads(str(data['metadata']))
    print(path.relative_to(root), data['keypoints'].shape, metadata['pose_providers'])
PY
```

预期每个文件的 shape 为 `(32, 134, 3)`，`pose_providers` 包含 `CUDAExecutionProvider`。

## 正式运行

仅处理 `test/linfen`：

```bash
cd /home/swq/Code/dwpose-video
conda run -n dwpose python batch_process_videos_ultra.py \
  --input /home/swq/Code/ePBSI/data/action_clips_4s/test/linfen \
  --output /home/swq/Code/ePBSI/data/action_clips_4s_keypoint/test/linfen \
  --model-dir /home/swq/models \
  --device cuda \
  --frames-per-clip 32 \
  --batch-size 8 \
  --no-video \
  --format npz \
  --direct-keypoint-output \
  --skip-existing
```

处理完整 `train/val/test`：

```bash
cd /home/swq/Code/dwpose-video
conda run -n dwpose python batch_process_videos_ultra.py \
  --input /home/swq/Code/ePBSI/data/action_clips_4s \
  --output /home/swq/Code/ePBSI/data/action_clips_4s_keypoint \
  --model-dir /home/swq/models \
  --device cuda \
  --frames-per-clip 32 \
  --batch-size 8 \
  --no-video \
  --format npz \
  --direct-keypoint-output \
  --skip-existing
```

`--skip-existing` 只按目标 `.npz` 是否存在跳过，不会覆盖已有结果。
