import cv2
import numpy as np

import onnxruntime

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def demo_postprocess(outputs, img_size, p6=False):
    """YOLOX 输出后处理：解码 bbox 坐标"""
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    # 防止 exp 溢出：clip 到合理范围
    outputs[..., 2:4] = np.exp(np.clip(outputs[..., 2:4], -10, 10)) * expanded_strides

    return outputs

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def inference_detector(session, oriImg):
    """单图检测推理"""
    input_shape = (640, 640)
    img, ratio = preprocess(oriImg, input_shape)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        isscore = final_scores > 0.3
        iscat = final_cls_inds == 0
        isbbox = [i and j for (i, j) in zip(isscore, iscat)]
        final_boxes = final_boxes[isbbox]
    else:
        final_boxes = np.array([])

    return final_boxes


def preprocess_batch(images, input_shape):
    """批量预处理多张图像

    Args:
        images: List[np.ndarray]，每张图像 shape (H, W, 3)
        input_shape: 目标尺寸 (H, W)

    Returns:
        batch_imgs: np.ndarray, shape (N, 3, H, W)
        ratios: List[float]，每张图的缩放比例
    """
    batch_imgs = []
    ratios = []

    for img in images:
        processed, ratio = preprocess(img, input_shape)
        batch_imgs.append(processed)
        ratios.append(ratio)

    return np.stack(batch_imgs, axis=0), ratios


def postprocess_single(predictions, ratio, nms_thr=0.45, score_thr=0.1, conf_thr=0.3):
    """单张图像的检测后处理（NMS + 过滤）

    Args:
        predictions: shape (8400, 85)，单张图的预测结果
        ratio: 预处理时的缩放比例

    Returns:
        final_boxes: np.ndarray，检测框坐标
    """
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        isscore = final_scores > conf_thr
        iscat = final_cls_inds == 0  # 只保留人类
        isbbox = [i and j for (i, j) in zip(isscore, iscat)]
        final_boxes = final_boxes[isbbox]
    else:
        final_boxes = np.array([])

    return final_boxes


def inference_detector_batch(session, images):
    """批量检测推理（需要动态 batch 模型）

    Args:
        session: ONNX Runtime session（需加载 yolox_l_dynamic.onnx）
        images: List[np.ndarray]，多张图像

    Returns:
        List[np.ndarray]：每张图的检测框列表
    """
    if len(images) == 0:
        return []

    input_shape = (640, 640)

    # 批量预处理
    batch_imgs, ratios = preprocess_batch(images, input_shape)

    # 单次批量推理
    ort_inputs = {session.get_inputs()[0].name: batch_imgs}
    output = session.run(None, ort_inputs)  # shape: (N, 8400, 85)

    # 批量解码（向量化）
    predictions = demo_postprocess(output[0].copy(), input_shape)  # (N, 8400, 85)

    # 逐图后处理（NMS 无法批量化）
    results = []
    for i in range(len(images)):
        final_boxes = postprocess_single(predictions[i], ratios[i])
        results.append(final_boxes)

    return results
