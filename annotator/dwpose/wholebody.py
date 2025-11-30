import cv2
import numpy as np
from typing import List, Tuple

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose, preprocess, inference_batch, postprocess_batch

class Wholebody:
    def __init__(self):
        device = 'cuda:0'
        providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
        onnx_det = 'models/yolox_l.onnx'
        onnx_pose = 'models/dw-ll_ucoco_384.onnx'

        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)

        # 获取姿态模型输入尺寸
        h, w = self.session_pose.get_inputs()[0].shape[2:]
        self.pose_input_size = (w, h)

    def __call__(self, oriImg):
        """单图推理（兼容原接口）"""
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints, scores = self._postprocess_keypoints(keypoints, scores)
        return keypoints, scores

    def _postprocess_keypoints(self, keypoints, scores):
        """关键点后处理：添加 neck、重排序"""
        if len(keypoints) == 0:
            return keypoints, scores

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]

        return keypoints, scores

    def batch_inference(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """批量推理多帧图像.

        利用姿态模型的动态 batch 支持，合并多帧中检测到的所有人进行批量推理。
        检测模型仍需逐帧执行（模型限制）。

        Args:
            images: 图像列表，每个 shape 为 (H, W, 3)

        Returns:
            结果列表: [(keypoints, scores), ...]，与输入图像一一对应
        """
        if len(images) == 0:
            return []

        # 阶段1：逐帧检测（检测模型不支持 batch）
        all_bboxes = []
        for img in images:
            bboxes = inference_detector(self.session_det, img)
            all_bboxes.append(bboxes)

        # 阶段2：收集所有人的 crops 和元信息
        all_crops = []
        all_centers = []
        all_scales = []
        crop_to_frame = []  # 记录每个 crop 属于哪一帧

        for frame_idx, (img, bboxes) in enumerate(zip(images, all_bboxes)):
            if len(bboxes) == 0:
                # 无检测结果时使用全图
                bboxes = [[0, 0, img.shape[1], img.shape[0]]]

            crops, centers, scales = preprocess(img, bboxes, self.pose_input_size)
            for crop, center, scale in zip(crops, centers, scales):
                all_crops.append(crop)
                all_centers.append(center)
                all_scales.append(scale)
                crop_to_frame.append(frame_idx)

        # 阶段3：批量姿态推理
        if len(all_crops) > 0:
            simcc_x, simcc_y = inference_batch(self.session_pose, all_crops)
            all_keypoints, all_scores = postprocess_batch(
                simcc_x, simcc_y, self.pose_input_size, all_centers, all_scales
            )
        else:
            all_keypoints, all_scores = np.array([]), np.array([])

        # 阶段4：按帧分组结果
        results = []
        crop_idx = 0
        for frame_idx in range(len(images)):
            frame_keypoints = []
            frame_scores = []

            while crop_idx < len(crop_to_frame) and crop_to_frame[crop_idx] == frame_idx:
                frame_keypoints.append(all_keypoints[crop_idx])
                frame_scores.append(all_scores[crop_idx])
                crop_idx += 1

            if len(frame_keypoints) > 0:
                keypoints = np.array(frame_keypoints)
                scores = np.array(frame_scores)
                keypoints, scores = self._postprocess_keypoints(keypoints, scores)
            else:
                keypoints = np.array([])
                scores = np.array([])

            results.append((keypoints, scores))

        return results


