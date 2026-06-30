#!/usr/bin/env python3
"""
DWPose 批量视频处理脚本 - 超级优化版

主要优化:
1. 批量推理（Batch Inference）- 一次处理多帧，利用姿态模型动态 batch
2. 多进程并行 - 多个视频文件并行处理
3. 可配置分辨率缩放
4. 跳帧处理选项
5. GPU 使用监控
6. 详细性能统计

作者: AI Assistant
日期: 2025-11-19
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotator.dwpose import DWposeDetector


KEYPOINT_COUNT = 134
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}


def build_frame_indices(total_frames: int, skip_frames: int, frames_per_clip: Optional[int]) -> List[int]:
    """生成需要读取的帧序号；frames_per_clip 存在时固定均匀采样。"""
    if total_frames <= 0:
        return []

    if frames_per_clip is None:
        return list(range(0, total_frames, skip_frames))

    if frames_per_clip < 1:
        raise ValueError("frames_per_clip 必须 >= 1")

    return np.linspace(0, total_frames - 1, frames_per_clip).round().astype(int).tolist()


def select_main_person_keypoints(candidate: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """从单帧多人体结果中选择置信度最高的人，输出固定关键点矩阵。"""
    fixed = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    if not isinstance(candidate, np.ndarray) or candidate.size == 0 or candidate.ndim != 3:
        return fixed

    person_count, keypoint_count, _locs = candidate.shape
    if person_count == 0:
        return fixed

    usable_keypoints = min(KEYPOINT_COUNT, keypoint_count)
    if isinstance(subset, np.ndarray) and subset.ndim == 2 and subset.shape[0] > 0:
        usable_people = min(person_count, subset.shape[0])
        usable_scores = min(usable_keypoints, subset.shape[1])
        scores = np.nan_to_num(subset[:usable_people, :usable_scores], nan=0.0)
        person_index = int(np.argmax(scores.sum(axis=1)))
        fixed[:usable_scores, 2] = scores[person_index]
    else:
        person_index = 0
        usable_scores = usable_keypoints
        fixed[:usable_scores, 2] = 1.0

    fixed[:usable_keypoints, :2] = np.nan_to_num(
        candidate[person_index, :usable_keypoints, :2], nan=0.0
    )
    return fixed


def build_fixed_keypoint_sequence(candidates: List[np.ndarray], subsets: List[np.ndarray]) -> np.ndarray:
    """把逐帧 DWPose 结果整理为 shape=(frames, 134, 3) 的固定序列。"""
    frames = [
        select_main_person_keypoints(candidate, subset)
        for candidate, subset in zip(candidates, subsets)
    ]
    if not frames:
        return np.zeros((0, KEYPOINT_COUNT, 3), dtype=np.float32)
    return np.stack(frames).astype(np.float32)


class DWposeDetectorBatch(DWposeDetector):
    """支持批量推理的 DWPose 检测器"""

    def __init__(
        self,
        use_dynamic_det: bool = False,
        model_dir: str = 'models',
        device: str = 'auto'
    ):
        self.use_dynamic_det = use_dynamic_det
        self.model_dir = model_dir
        self.device = device

        mode_str = "动态batch检测" if use_dynamic_det else "标准检测"
        print(f"🚀 初始化 DWPose 检测器（{mode_str} + 批量姿态推理）...")

        # 重新初始化 Wholebody 以使用正确的检测模型
        from annotator.dwpose.wholebody import Wholebody
        self.pose_estimation = Wholebody(
            use_dynamic_det=use_dynamic_det,
            model_dir=model_dir,
            device=device
        )

        det_providers = self.pose_estimation.det_providers
        pose_providers = self.pose_estimation.pose_providers
        if 'CUDAExecutionProvider' in det_providers and 'CUDAExecutionProvider' in pose_providers:
            print("✅ GPU 加速已实际启用（CUDAExecutionProvider）")
        else:
            print("⚠️  当前模型 session 未实际启用 CUDA，使用 CPU provider")
        print(f"   det providers: {det_providers}")
        print(f"   pose providers: {pose_providers}")
    
    def __call__(self, oriImg):
        """
        处理单帧图像

        Args:
            oriImg: 输入图像 (H, W, 3)

        Returns:
            canvas: 可视化图像
            candidate: 关键点坐标 (N, K, 3)，原始像素坐标
            subset: 关键点置信度 (N, K)
        """
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape

        # 直接调用 Wholebody 获取关键点（返回原始像素坐标）
        candidate, subset = self.pose_estimation(oriImg)

        # 复制一份用于可视化（需要归一化坐标）
        candidate_vis = candidate.copy()
        nums, keys, locs = candidate_vis.shape
        candidate_vis[..., 0] /= float(W)
        candidate_vis[..., 1] /= float(H)

        # 构建可视化所需的数据结构
        body = candidate_vis[:, :18].copy()
        body = body.reshape(nums * 18, locs)
        score = subset[:, :18].copy()

        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1

        un_visible = subset < 0.3
        candidate_vis[un_visible] = -1

        foot = candidate_vis[:, 18:24]
        faces = candidate_vis[:, 24:92]
        hands = candidate_vis[:, 92:113]
        hands = np.vstack([hands, candidate_vis[:, 113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        # 生成可视化图像
        from annotator.dwpose import draw_pose
        canvas = draw_pose(pose, H, W)

        return canvas, candidate, subset
    
    def process_batch(self, frames: List[np.ndarray]) -> List[Tuple]:
        """
        批量处理多帧（真正的批量推理：合并多帧多人的 crops 进行批量姿态估计）

        Args:
            frames: 帧列表，每个 shape 为 (H, W, 3)

        Returns:
            结果列表 [(canvas, candidate, subset), ...]
        """
        if len(frames) == 0:
            return []

        # 使用 Wholebody 的批量推理接口
        batch_results = self.pose_estimation.batch_inference(frames)

        # 为每帧生成可视化画布
        results = []
        for frame, (candidate, subset) in zip(frames, batch_results):
            H, W, C = frame.shape

            if len(candidate) == 0:
                # 无检测结果时返回空画布
                canvas = np.zeros((H, W, 3), dtype=np.uint8)
                results.append((canvas, candidate, subset))
                continue

            # 复制一份用于可视化（需要归一化坐标）
            candidate_vis = candidate.copy()
            nums, keys, locs = candidate_vis.shape
            candidate_vis[..., 0] /= float(W)
            candidate_vis[..., 1] /= float(H)

            # 构建可视化所需的数据结构
            body = candidate_vis[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18].copy()

            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate_vis[un_visible] = -1

            foot = candidate_vis[:, 18:24]
            faces = candidate_vis[:, 24:92]
            hands = candidate_vis[:, 92:113]
            hands = np.vstack([hands, candidate_vis[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # 生成可视化图像
            from annotator.dwpose import draw_pose
            canvas = draw_pose(pose, H, W)

            results.append((canvas, candidate, subset))

        return results


class BatchVideoProcessorUltra:
    """超级优化的批量视频处理器"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: int = 4,
        scale_factor: float = 1.0,
        skip_frames: int = 1,
        save_video: bool = True,
        save_format: str = 'npz',
        skip_existing: bool = False,
        use_dynamic_det: bool = False,
        frames_per_clip: Optional[int] = None,
        model_dir: str = 'models',
        device: str = 'auto',
        direct_keypoint_output: bool = False,
        limit: Optional[int] = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.skip_frames = skip_frames
        self.save_video = save_video
        self.save_format = save_format
        self.skip_existing = skip_existing
        self.use_dynamic_det = use_dynamic_det
        self.frames_per_clip = frames_per_clip
        self.model_dir = model_dir
        self.device = device
        self.direct_keypoint_output = direct_keypoint_output
        self.limit = limit

        # 创建输出目录
        self.video_output_dir = self.output_dir / 'video_output'
        self.keypoints_output_dir = (
            self.output_dir if direct_keypoint_output else self.output_dir / 'keypoints_output'
        )

        if self.save_video:
            self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keypoints_output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化检测器
        self.detector = DWposeDetectorBatch(
            use_dynamic_det=use_dynamic_det,
            model_dir=model_dir,
            device=device
        )

        print(f"\n📊 配置信息:")
        print(f"  - 批量大小: {self.batch_size}")
        print(f"  - 分辨率缩放: {self.scale_factor:.2f}x")
        print(f"  - 跳帧间隔: {self.skip_frames}")
        print(f"  - 每段固定帧数: {self.frames_per_clip}")
        print(f"  - 保存视频: {self.save_video}")
        print(f"  - 保存格式: {self.save_format}")
        print(f"  - 跳过已存在: {self.skip_existing}")
        print(f"  - 动态batch检测: {use_dynamic_det}")
        print(f"  - 模型目录: {self.model_dir}")
        print(f"  - 请求设备: {self.device}")
        print(f"  - 直接输出关键点树: {self.direct_keypoint_output}")
        print(f"  - 处理数量限制: {self.limit}")
    
    def find_all_videos(self) -> List[Path]:
        """递归查找所有视频文件"""
        videos = []

        for ext in VIDEO_EXTENSIONS:
            videos.extend(self.input_dir.rglob(f'*{ext}'))

        return sorted(videos)

    def get_output_paths(self, video_path: Path) -> Tuple[Optional[Path], Path]:
        """获取输出路径"""
        rel_path = video_path.relative_to(self.input_dir)

        # 视频输出路径
        if self.save_video:
            video_output = self.video_output_dir / rel_path
            video_output.parent.mkdir(parents=True, exist_ok=True)
        else:
            video_output = None

        # 关键点输出路径
        keypoints_output = self.keypoints_output_dir / rel_path.with_suffix(f'.{self.save_format}')
        keypoints_output.parent.mkdir(parents=True, exist_ok=True)

        return video_output, keypoints_output

    def should_skip(self, video_output: Optional[Path], keypoints_output: Path) -> bool:
        """检查是否应该跳过"""
        if not self.skip_existing:
            return False

        if self.save_video and video_output and not video_output.exists():
            return False

        if not keypoints_output.exists():
            return False

        return True

    def process_single_video(self, video_path: Path) -> Dict:
        """处理单个视频（批量推理版本）"""
        video_name = video_path.name
        start_time = time.time()

        try:
            # 获取输出路径
            video_output, keypoints_output = self.get_output_paths(video_path)

            # 检查是否跳过
            if self.should_skip(video_output, keypoints_output):
                return {
                    'video': str(video_path),
                    'status': 'skipped',
                    'reason': 'already exists'
                }

            # 打开视频
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 计算缩放后的尺寸
            scaled_width = int(width * self.scale_factor)
            scaled_height = int(height * self.scale_factor)

            # 初始化视频写入器
            if self.save_video and video_output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    str(video_output),
                    fourcc,
                    fps,
                    (width, height)  # 输出原始分辨率
                )
            else:
                out = None

            # 存储关键点数据
            all_candidates = []
            all_subsets = []

            # 批量处理帧
            frame_indices = build_frame_indices(
                total_frames=total_frames,
                skip_frames=self.skip_frames,
                frames_per_clip=self.frames_per_clip
            )
            processed_frames = 0

            pbar = tqdm(total=len(frame_indices), desc=f"处理 {video_name}")

            for batch_start in range(0, len(frame_indices), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(frame_indices))
                batch_indices = frame_indices[batch_start:batch_end]

                # 读取批量帧
                frames = []
                for idx in batch_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 缩放帧
                    if self.scale_factor != 1.0:
                        frame = cv2.resize(frame, (scaled_width, scaled_height))

                    frames.append(frame)

                if not frames:
                    break

                # 批量推理
                results = self.detector.process_batch(frames)

                # 处理结果
                for i, (vis_frame, candidate, subset) in enumerate(results):
                    # 如果缩放了，需要将关键点坐标缩放回原始尺寸
                    if self.scale_factor != 1.0:
                        if candidate is not None and len(candidate) > 0:
                            candidate = candidate.copy()
                            candidate[:, :, :2] /= self.scale_factor

                        # 将可视化帧也缩放回原始尺寸
                        vis_frame = cv2.resize(vis_frame, (width, height))

                    # 保存可视化视频
                    if out is not None:
                        out.write(vis_frame)

                    # 保存关键点
                    all_candidates.append(candidate if candidate is not None else np.array([]))
                    all_subsets.append(subset if subset is not None else np.array([]))

                    processed_frames += 1
                    pbar.update(1)

            pbar.close()
            cap.release()
            if out is not None:
                out.release()

            keypoints = build_fixed_keypoint_sequence(all_candidates, all_subsets)
            det_providers = self.detector.pose_estimation.det_providers
            pose_providers = self.detector.pose_estimation.pose_providers

            # 保存关键点数据
            metadata = {
                'video_path': str(video_path),
                'relative_path': str(video_path.relative_to(self.input_dir)),
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'frame_indices': frame_indices[:processed_frames],
                'original_resolution': (width, height),
                'scale_factor': self.scale_factor,
                'skip_frames': self.skip_frames,
                'frames_per_clip': self.frames_per_clip,
                'keypoint_count': KEYPOINT_COUNT,
                'sequence_shape': list(keypoints.shape),
                'keypoint_layout': 'DWPose wholebody with inserted neck',
                'model_dir': self.model_dir,
                'det_model_path': self.detector.pose_estimation.det_model_path,
                'pose_model_path': self.detector.pose_estimation.pose_model_path,
                'device': self.device,
                'requested_providers': self.detector.pose_estimation.requested_providers,
                'det_providers': det_providers,
                'pose_providers': pose_providers,
                'format': self.save_format,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            if self.save_format == 'npz':
                self.save_keypoints_npz(keypoints_output, keypoints, metadata)
            else:
                self.save_keypoints_json(keypoints_output, keypoints, metadata)

            # 计算性能统计
            processing_time = time.time() - start_time
            processing_fps = processed_frames / processing_time if processing_time > 0 else 0

            return {
                'video': str(video_path),
                'status': 'success',
                'frames': processed_frames,
                'processing_time': processing_time,
                'fps': processing_fps,
                'video_output': str(video_output) if video_output else None,
                'keypoints_output': str(keypoints_output)
            }

        except Exception as e:
            import traceback
            return {
                'video': str(video_path),
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def save_keypoints_npz(self, output_path: Path, keypoints: np.ndarray, metadata: Dict):
        """保存 NPZ 格式的关键点数据"""
        np.savez_compressed(
            output_path,
            keypoints=keypoints.astype(np.float32),
            frame_indices=np.asarray(metadata.get('frame_indices', []), dtype=np.int32),
            metadata=np.array(json.dumps(metadata, ensure_ascii=False))
        )

    def save_keypoints_json(self, output_path: Path, keypoints: np.ndarray, metadata: Dict):
        """保存 JSON 格式的关键点数据"""
        data = {
            **metadata,
            'keypoints': keypoints.tolist()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def process_all_videos(self):
        """批量处理所有视频"""
        videos = self.find_all_videos()
        if self.limit is not None:
            videos = videos[:self.limit]

        if not videos:
            print(f"❌ 在 {self.input_dir} 中未找到视频文件")
            return

        print(f"\n📹 找到 {len(videos)} 个视频文件")
        print(f"📂 输入目录: {self.input_dir}")
        print(f"📂 输出目录: {self.output_dir}")

        results = []
        success_count = 0
        failed_count = 0
        skipped_count = 0
        total_frames = 0
        total_time = 0

        for i, video_path in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] 处理: {video_path.name}")

            result = self.process_single_video(video_path)
            results.append(result)

            if result['status'] == 'success':
                success_count += 1
                total_frames += result['frames']
                total_time += result['processing_time']
                print(f"  ✅ 成功")
                print(f"     - 帧数: {result['frames']}")
                print(f"     - 处理时间: {result['processing_time']:.2f}s")
                print(f"     - 处理速度: {result['fps']:.2f} fps")
                if result['video_output']:
                    print(f"     - 视频输出: {Path(result['video_output']).name}")
                print(f"     - 关键点输出: {Path(result['keypoints_output']).name}")
            elif result['status'] == 'skipped':
                skipped_count += 1
                print(f"  ⏭️  跳过（已存在）")
            else:
                failed_count += 1
                print(f"  ❌ 失败: {result['error']}")

        # 生成报告
        avg_fps = total_frames / total_time if total_time > 0 else 0

        report = {
            'total': len(videos),
            'success': success_count,
            'skipped': skipped_count,
            'failed': failed_count,
            'total_frames_processed': total_frames,
            'total_processing_time': total_time,
            'average_fps': avg_fps,
            'configuration': {
                'batch_size': self.batch_size,
                'scale_factor': self.scale_factor,
                'skip_frames': self.skip_frames,
                'frames_per_clip': self.frames_per_clip,
                'save_video': self.save_video,
                'save_format': self.save_format,
                'model_dir': self.model_dir,
                'device': self.device,
                'direct_keypoint_output': self.direct_keypoint_output,
                'limit': self.limit
            },
            'results': results
        }

        report_path = self.output_dir / 'processing_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印总结
        print("\n" + "=" * 80)
        print("处理完成")
        print("=" * 80)
        print(f"总计: {len(videos)} 个视频")
        print(f"成功: {success_count}")
        print(f"跳过: {skipped_count}")
        print(f"失败: {failed_count}")
        print(f"总帧数: {total_frames}")
        print(f"总时间: {total_time:.2f}s")
        print(f"平均速度: {avg_fps:.2f} fps")
        print(f"\n📊 处理报告: {report_path}")
        print("=" * 80)


def _worker_init(
    use_dynamic_det: bool = False,
    model_dir: str = 'models',
    device: str = 'auto'
):
    """多进程 worker 初始化函数：每个进程初始化自己的检测器"""
    global _worker_detector
    _worker_detector = DWposeDetectorBatch(
        use_dynamic_det=use_dynamic_det,
        model_dir=model_dir,
        device=device
    )


def _worker_process_video(video_info: Dict, config: Dict) -> Dict:
    """多进程 worker 处理单个视频的函数

    Args:
        video_info: 包含 video_path, input_dir 的字典
        config: 处理配置参数

    Returns:
        处理结果字典
    """
    global _worker_detector

    video_path = Path(video_info['video_path'])
    input_dir = Path(video_info['input_dir'])
    output_dir = Path(config['output_dir'])

    # 获取输出路径
    rel_path = video_path.relative_to(input_dir)

    video_output_dir = output_dir / 'video_output'
    keypoints_output_dir = (
        output_dir if config.get('direct_keypoint_output') else output_dir / 'keypoints_output'
    )

    if config['save_video']:
        video_output = video_output_dir / rel_path
        video_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_output = None

    keypoints_output = keypoints_output_dir / rel_path.with_suffix(f".{config['save_format']}")
    keypoints_output.parent.mkdir(parents=True, exist_ok=True)

    # 检查是否跳过
    if config['skip_existing']:
        if keypoints_output.exists():
            if not config['save_video'] or (video_output and video_output.exists()):
                return {
                    'video': str(video_path),
                    'status': 'skipped',
                    'reason': 'already exists'
                }

    start_time = time.time()

    try:
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale_factor = config['scale_factor']
        scaled_width = int(width * scale_factor)
        scaled_height = int(height * scale_factor)

        # 初始化视频写入器
        if config['save_video'] and video_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))
        else:
            out = None

        all_candidates = []
        all_subsets = []

        frame_indices = build_frame_indices(
            total_frames=total_frames,
            skip_frames=config['skip_frames'],
            frames_per_clip=config.get('frames_per_clip')
        )
        processed_frames = 0
        batch_size = config['batch_size']

        for batch_start in range(0, len(frame_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(frame_indices))
            batch_indices = frame_indices[batch_start:batch_end]

            frames = []
            for idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, (scaled_width, scaled_height))
                frames.append(frame)

            if not frames:
                break

            # 批量推理
            results = _worker_detector.process_batch(frames)

            for vis_frame, candidate, subset in results:
                if scale_factor != 1.0:
                    if candidate is not None and len(candidate) > 0:
                        candidate = candidate.copy()
                        candidate[:, :, :2] /= scale_factor
                    vis_frame = cv2.resize(vis_frame, (width, height))

                if out is not None:
                    out.write(vis_frame)

                all_candidates.append(candidate if candidate is not None else np.array([]))
                all_subsets.append(subset if subset is not None else np.array([]))
                processed_frames += 1

        cap.release()
        if out is not None:
            out.release()

        keypoints = build_fixed_keypoint_sequence(all_candidates, all_subsets)
        det_providers = _worker_detector.pose_estimation.det_providers
        pose_providers = _worker_detector.pose_estimation.pose_providers

        # 保存关键点数据
        metadata = {
            'video_path': str(video_path),
            'relative_path': str(video_path.relative_to(input_dir)),
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'frame_indices': frame_indices[:processed_frames],
            'original_resolution': (width, height),
            'scale_factor': scale_factor,
            'skip_frames': config['skip_frames'],
            'frames_per_clip': config.get('frames_per_clip'),
            'keypoint_count': KEYPOINT_COUNT,
            'sequence_shape': list(keypoints.shape),
            'keypoint_layout': 'DWPose wholebody with inserted neck',
            'model_dir': config.get('model_dir'),
            'det_model_path': _worker_detector.pose_estimation.det_model_path,
            'pose_model_path': _worker_detector.pose_estimation.pose_model_path,
            'device': config.get('device'),
            'requested_providers': _worker_detector.pose_estimation.requested_providers,
            'det_providers': det_providers,
            'pose_providers': pose_providers,
            'format': config['save_format'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        if config['save_format'] == 'npz':
            np.savez_compressed(
                keypoints_output,
                keypoints=keypoints.astype(np.float32),
                frame_indices=np.asarray(metadata.get('frame_indices', []), dtype=np.int32),
                metadata=np.array(json.dumps(metadata, ensure_ascii=False))
            )
        else:
            with open(keypoints_output, 'w', encoding='utf-8') as f:
                json.dump({**metadata, 'keypoints': keypoints.tolist()}, f, indent=2, ensure_ascii=False)

        processing_time = time.time() - start_time

        return {
            'video': str(video_path),
            'status': 'success',
            'frames': processed_frames,
            'processing_time': processing_time,
            'fps': processed_frames / processing_time if processing_time > 0 else 0,
            'video_output': str(video_output) if video_output else None,
            'keypoints_output': str(keypoints_output)
        }

    except Exception as e:
        import traceback
        return {
            'video': str(video_path),
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def process_videos_multiprocess(videos: List[Path], input_dir: Path, config: Dict, num_workers: int):
    """使用多进程并行处理视频

    Args:
        videos: 视频文件路径列表
        input_dir: 输入根目录
        config: 处理配置（包含 use_dynamic_det）
        num_workers: 进程数
    """
    use_dynamic_det = config.get('use_dynamic_det', False)
    mode_str = "动态batch检测" if use_dynamic_det else "标准检测"
    print(f"\n🚀 启动 {num_workers} 个并行进程处理 {len(videos)} 个视频（{mode_str}）...")

    # 准备任务列表
    video_infos = [{'video_path': str(v), 'input_dir': str(input_dir)} for v in videos]

    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_frames = 0
    total_time = 0

    # 使用进程池（传递 use_dynamic_det 给 worker 初始化函数）
    init_fn = partial(
        _worker_init,
        use_dynamic_det=use_dynamic_det,
        model_dir=config.get('model_dir', 'models'),
        device=config.get('device', 'auto')
    )
    with mp.Pool(processes=num_workers, initializer=init_fn) as pool:
        worker_fn = partial(_worker_process_video, config=config)

        # 使用 imap_unordered 获取结果并显示进度
        with tqdm(total=len(videos), desc="处理进度") as pbar:
            for result in pool.imap_unordered(worker_fn, video_infos):
                results.append(result)

                if result['status'] == 'success':
                    success_count += 1
                    total_frames += result['frames']
                    total_time += result['processing_time']
                    pbar.set_postfix({'成功': success_count, '速度': f"{result['fps']:.1f} fps"})
                elif result['status'] == 'skipped':
                    skipped_count += 1
                else:
                    failed_count += 1
                    print(f"\n  ❌ 失败: {Path(result['video']).name}: {result.get('error', 'Unknown')}")

                pbar.update(1)

    # 生成报告
    avg_fps = total_frames / total_time if total_time > 0 else 0
    output_dir = Path(config['output_dir'])

    report = {
        'total': len(videos),
        'success': success_count,
        'skipped': skipped_count,
        'failed': failed_count,
        'total_frames_processed': total_frames,
        'total_processing_time': total_time,
        'average_fps': avg_fps,
        'num_workers': num_workers,
        'configuration': config,
        'results': results
    }

    report_path = output_dir / 'processing_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"总计: {len(videos)} 个视频")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"总帧数: {total_frames}")
    print(f"总处理时间: {total_time:.2f}s")
    print(f"平均速度: {avg_fps:.2f} fps")
    print(f"并行进程数: {num_workers}")
    print(f"\n📊 处理报告: {report_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='DWPose 批量视频处理 - 超级优化版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
优化特性:
  1. 批量推理 - 一次处理多帧，利用姿态模型动态 batch
  2. 多进程并行 - 多个视频文件并行处理（--num-workers）
  3. 分辨率缩放 - 降低输入分辨率，加快处理速度
  4. 跳帧处理 - 只处理关键帧，大幅提升速度
  5. GPU 监控 - 自动检测 GPU 使用情况

使用示例:
  # 基本使用（单进程，批量大小 4）
  python batch_process_videos_ultra.py \\
      --input ../data/UCF-101/ApplyEyeMakeup \\
      --output ../data/dwpose

  # 多进程并行（4 进程）
  python batch_process_videos_ultra.py \\
      --input ../data/UCF-101 \\
      --output ../data/dwpose \\
      --num-workers 4

  # 高性能配置（批量 8 + 缩放 0.75x + 多进程）
  python batch_process_videos_ultra.py \\
      --input ../data/UCF-101 \\
      --output ../data/dwpose \\
      --batch-size 8 \\
      --scale 0.75 \\
      --num-workers 2

  # 极速模式（仅关键点，跳帧）
  python batch_process_videos_ultra.py \\
      --input ../data/UCF-101 \\
      --output ../data/dwpose \\
      --batch-size 16 \\
      --scale 0.5 \\
      --skip-frames 2 \\
      --no-video
        """
    )

    parser.add_argument('--input', type=str, required=True, help='输入视频目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--batch-size', type=int, default=4, help='帧批量大小（默认: 4）')
    parser.add_argument('--scale', type=float, default=1.0, help='分辨率缩放因子（默认: 1.0）')
    parser.add_argument('--skip-frames', type=int, default=1, help='跳帧间隔（默认: 1，不跳帧）')
    parser.add_argument('--frames-per-clip', type=int, default=None,
                        help='每个视频固定均匀采样的帧数；设置后优先于 --skip-frames')
    parser.add_argument('--no-video', action='store_true', help='不保存可视化视频')
    parser.add_argument('--format', type=str, default='npz', choices=['npz', 'json'], help='关键点保存格式')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已处理的文件')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='并行进程数（默认: 1，单进程）。注意：多进程模式下每个进程都会加载模型，需要更多显存')
    parser.add_argument('--use-dynamic-det', action='store_true',
                        help='使用动态 batch 检测模型（需要 models/yolox_l_dynamic.onnx）')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='DWPose ONNX 模型目录（默认: models）')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='推理设备（默认: auto）')
    parser.add_argument('--direct-keypoint-output', action='store_true',
                        help='直接在 --output 下按输入相对路径写关键点文件，不额外创建 keypoints_output')
    parser.add_argument('--limit', type=int, default=None,
                        help='仅处理排序后的前 N 个视频，用于 smoke 验证')

    args = parser.parse_args()

    # 验证参数
    if args.batch_size < 1:
        print("❌ 错误: batch-size 必须 >= 1")
        return 1

    if args.scale <= 0 or args.scale > 1.0:
        print("❌ 错误: scale 必须在 (0, 1.0] 范围内")
        return 1

    if args.skip_frames < 1:
        print("❌ 错误: skip-frames 必须 >= 1")
        return 1

    if args.frames_per_clip is not None and args.frames_per_clip < 1:
        print("❌ 错误: frames-per-clip 必须 >= 1")
        return 1

    if args.num_workers < 1:
        print("❌ 错误: num-workers 必须 >= 1")
        return 1

    if args.limit is not None and args.limit < 1:
        print("❌ 错误: limit 必须 >= 1")
        return 1

    # 根据进程数选择处理模式
    if args.num_workers == 1:
        # 单进程模式（原有逻辑）
        processor = BatchVideoProcessorUltra(
            input_dir=args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
            scale_factor=args.scale,
            skip_frames=args.skip_frames,
            save_video=not args.no_video,
            save_format=args.format,
            skip_existing=args.skip_existing,
            use_dynamic_det=args.use_dynamic_det,
            frames_per_clip=args.frames_per_clip,
            model_dir=args.model_dir,
            device=args.device,
            direct_keypoint_output=args.direct_keypoint_output,
            limit=args.limit
        )
        processor.process_all_videos()
    else:
        # 多进程模式
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_video:
            (output_dir / 'video_output').mkdir(parents=True, exist_ok=True)
        if not args.direct_keypoint_output:
            (output_dir / 'keypoints_output').mkdir(parents=True, exist_ok=True)

        # 查找所有视频
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(input_dir.rglob(f'*{ext}'))
        videos = sorted(videos)
        if args.limit is not None:
            videos = videos[:args.limit]

        if not videos:
            print(f"❌ 在 {input_dir} 中未找到视频文件")
            return 1

        config = {
            'output_dir': str(output_dir),
            'batch_size': args.batch_size,
            'scale_factor': args.scale,
            'skip_frames': args.skip_frames,
            'frames_per_clip': args.frames_per_clip,
            'save_video': not args.no_video,
            'save_format': args.format,
            'skip_existing': args.skip_existing,
            'use_dynamic_det': args.use_dynamic_det,
            'model_dir': args.model_dir,
            'device': args.device,
            'direct_keypoint_output': args.direct_keypoint_output,
            'limit': args.limit
        }

        process_videos_multiprocess(videos, input_dir, config, args.num_workers)

    return 0


if __name__ == '__main__':
    # 多进程需要在 __main__ 中启动
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
