import argparse
import cv2
import numpy as np
import onnxruntime as ort


def softmax_channel(cls_score, num_anchors):
    """
    Reproduce MXNet:
        Reshape(cls_score, shape=(0, 2, -1, 0))
        SoftmaxActivation(mode="channel")
        Reshape back to [1, 2*num_anchors, H, W]

    cls_score: [1, 2A, H, W]
    """
    n, c, h, w = cls_score.shape

    cls_reshape = cls_score.reshape((n, 2, -1, w))
    cls_reshape = cls_reshape - np.max(cls_reshape, axis=1, keepdims=True)
    cls_exp = np.exp(cls_reshape)
    cls_prob = cls_exp / np.sum(cls_exp, axis=1, keepdims=True)

    cls_prob = cls_prob.reshape((n, 2 * num_anchors, h, w))
    return cls_prob


def generate_base_anchors(base_size=16, ratios=(1.0,), scales=(1.0,)):
    """
    Match common MXNet RetinaFace anchor generation style.
    Base anchor: [0, 0, base_size-1, base_size-1]
    """
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], dtype=np.float32)

    x1, y1, x2, y2 = base_anchor
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    x_ctr = x1 + 0.5 * (w - 1)
    y_ctr = y1 + 0.5 * (h - 1)
    size = w * h

    anchors = []

    for ratio in ratios:
        size_ratio = size / ratio
        ws = np.round(np.sqrt(size_ratio))
        hs = np.round(ws * ratio)

        for scale in scales:
            ww = ws * scale
            hh = hs * scale

            anchor = [
                x_ctr - 0.5 * (ww - 1),
                y_ctr - 0.5 * (hh - 1),
                x_ctr + 0.5 * (ww - 1),
                y_ctr + 0.5 * (hh - 1),
            ]
            anchors.append(anchor)

    return np.array(anchors, dtype=np.float32)


def generate_anchors_plane(feat_h, feat_w, stride, base_anchors):
    """
    Generate anchors for one FPN level.

    Output order:
        y -> x -> anchor
    """
    shift_x = np.arange(0, feat_w) * stride
    shift_y = np.arange(0, feat_h) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).transpose().astype(np.float32)

    a = base_anchors.shape[0]
    k = shifts.shape[0]

    anchors = (
        base_anchors.reshape((1, a, 4)) +
        shifts.reshape((k, 1, 4))
    )

    anchors = anchors.reshape((k * a, 4))
    return anchors.astype(np.float32)


def bbox_pred(anchors, bbox_deltas):
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)

    dx = bbox_deltas[:, 0]
    dy = bbox_deltas[:, 1]
    dw = bbox_deltas[:, 2]
    dh = bbox_deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros_like(bbox_deltas, dtype=np.float32)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes


def landmark_pred(anchors, landmark_deltas):
    """
    anchors: [N, 4]
    landmark_deltas: [N, 5, 2]
    """
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)

    pred = np.zeros_like(landmark_deltas, dtype=np.float32)

    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y

    return pred


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)

        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


class RetinaFaceONNX:
    def __init__(
        self,
        onnx_path,
        input_size=(640, 640),
        ctx_id=0,
        score_thresh=0.8,
        nms_thresh=0.3,
        cls_is_score=True,
    ):
        self.onnx_path = onnx_path
        self.input_size = input_size  # (w, h)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.cls_is_score = cls_is_score

        self.strides = [32, 16, 8]

        # Your final config uses RAC_SSH.
        self.anchor_cfg = {
            32: {"SCALES": (32, 16), "BASE_SIZE": 16, "RATIOS": (1.0,)},
            16: {"SCALES": (8, 4), "BASE_SIZE": 16, "RATIOS": (1.0,)},
            8: {"SCALES": (2, 1), "BASE_SIZE": 16, "RATIOS": (1.0,)},
        }

        if ctx_id >= 0:
            providers = [
                ("CUDAExecutionProvider", {"device_id": ctx_id}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        print("ONNX input:")
        for x in self.session.get_inputs():
            print(" ", x.name, x.shape, x.type)

        print("ONNX outputs:")
        for y in self.session.get_outputs():
            print(" ", y.name, y.shape, y.type)

    def preprocess(self, img):
        input_w, input_h = self.input_size
        orig_h, orig_w = img.shape[:2]

        scale = min(input_w / float(orig_w), input_h / float(orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(img, (new_w, new_h))
        resized = resized.astype(np.float32)

        canvas = np.zeros((input_h, input_w, 3), dtype=np.float32)

        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = canvas.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

        meta = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "new_w": new_w,
            "new_h": new_h,
            "orig_w": orig_w,
            "orig_h": orig_h,
        }
        return blob, meta

    def group_outputs(self, outputs):
        """
        Expected order from no-softmax export:
            stride32: cls_score, bbox_pred, landmark_pred
            stride16: cls_score, bbox_pred, landmark_pred
            stride8 : cls_score, bbox_pred, landmark_pred
        """
        grouped = {}

        # Try by name first.
        for name, arr in zip(self.output_names, outputs):
            lname = name.lower()

            for stride in self.strides:
                if f"stride{stride}" not in lname:
                    continue

                if "cls" in lname:
                    grouped[(stride, "cls")] = arr
                elif "bbox" in lname:
                    grouped[(stride, "bbox")] = arr
                elif "landmark" in lname:
                    grouped[(stride, "landmark")] = arr

        required = []
        for s in self.strides:
            required.extend([(s, "cls"), (s, "bbox"), (s, "landmark")])

        if all(k in grouped for k in required):
            return grouped

        # Fallback by fixed order.
        if len(outputs) != 9:
            raise RuntimeError(f"Expected 9 outputs, got {len(outputs)}")

        grouped = {}
        idx = 0
        for stride in self.strides:
            grouped[(stride, "cls")] = outputs[idx]
            grouped[(stride, "bbox")] = outputs[idx + 1]
            grouped[(stride, "landmark")] = outputs[idx + 2]
            idx += 3

        return grouped

    def decode_one_stride(self, cls, bbox, landmark, stride):
        cls = np.asarray(cls)
        bbox = np.asarray(bbox)
        landmark = np.asarray(landmark)

        if cls.ndim == 3:
            cls = cls[None, ...]
        if bbox.ndim == 3:
            bbox = bbox[None, ...]
        if landmark.ndim == 3:
            landmark = landmark[None, ...]

        _, cls_c, feat_h, feat_w = cls.shape

        cfg = self.anchor_cfg[stride]
        num_anchors = len(cfg["SCALES"]) * len(cfg["RATIOS"])

        if cls_c != 2 * num_anchors:
            raise RuntimeError(
                f"stride {stride}: cls channel error, got {cls_c}, expected {2 * num_anchors}"
            )

        base_anchors = generate_base_anchors(
            base_size=cfg["BASE_SIZE"],
            ratios=cfg["RATIOS"],
            scales=cfg["SCALES"],
        )

        anchors = generate_anchors_plane(
            feat_h=feat_h,
            feat_w=feat_w,
            stride=stride,
            base_anchors=base_anchors,
        )

        if self.cls_is_score:
            cls_prob = softmax_channel(cls, num_anchors)
        else:
            cls_prob = cls

        # Positive face probability.
        scores = cls_prob[:, num_anchors: 2 * num_anchors, :, :]
        scores = scores.transpose(0, 2, 3, 1).reshape(-1)

        bbox = bbox.reshape(1, num_anchors, 4, feat_h, feat_w)
        bbox = bbox.transpose(0, 3, 4, 1, 2).reshape(-1, 4)

        boxes = bbox_pred(anchors, bbox)

        landmark = landmark.reshape(1, num_anchors, 10, feat_h, feat_w)
        landmark = landmark.transpose(0, 3, 4, 1, 2).reshape(-1, 5, 2)

        landmarks = landmark_pred(anchors, landmark)

        input_w, input_h = self.input_size

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h - 1)

        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, input_w - 1)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, input_h - 1)

        return boxes, scores, landmarks

    def detect(self, img):
        original_h, original_w = img.shape[:2]
        input_w, input_h = self.input_size

        blob, meta = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})

        grouped = self.group_outputs(outputs)

        all_boxes = []
        all_scores = []
        all_landmarks = []

        for stride in self.strides:
            boxes, scores, landmarks = self.decode_one_stride(
                cls=grouped[(stride, "cls")],
                bbox=grouped[(stride, "bbox")],
                landmark=grouped[(stride, "landmark")],
                stride=stride,
            )

            keep = np.where(scores >= self.score_thresh)[0]
            if keep.size == 0:
                continue

            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_landmarks.append(landmarks[keep])

        if len(all_boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        landmarks = np.vstack(all_landmarks)

        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)

        keep = nms(dets, self.nms_thresh)
        dets = dets[keep]
        landmarks = landmarks[keep]

        # 从 letterbox/padding 后的输入坐标映射回原图坐标
        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]

        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale

        landmarks[:, :, 0] = (landmarks[:, :, 0] - pad_x) / scale
        landmarks[:, :, 1] = (landmarks[:, :, 1] - pad_y) / scale

        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, original_w - 1)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, original_h - 1)

        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, original_w - 1)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, original_h - 1)

        return dets.astype(np.float32), landmarks.astype(np.float32)


def draw_result(img, faces, landmarks, save_path):
    vis = img.copy()

    for i in range(faces.shape[0]):
        box = faces[i].astype(int)
        x1, y1, x2, y2, score = box[0], box[1], box[2], box[3], faces[i][4]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis,
            f"{score:.3f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        landmark5 = landmarks[i].astype(int)
        for j in range(5):
            color = (255, 0, 0)
            if j in [0, 1]:
                color = (0, 255, 0)
            elif j == 2:
                color = (0, 255, 255)
            elif j in [3, 4]:
                color = (0, 0, 255)

            cv2.circle(vis, (landmark5[j][0], landmark5[j][1]), 2, color, 2)

    cv2.imwrite(save_path, vis)
    print("saved:", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="/data/xxl/Word_model/insightface/detection/retinaface/model/retinaface.onnx")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--save", type=str, default="./retinaface_onnx_result1.jpg")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--score-thresh", type=float, default=0.8)
    parser.add_argument("--nms-thresh", type=float, default=0.3)
    parser.add_argument("--ctx-id", type=int, default=0, help="0 for GPU0, -1 for CPU")
    parser.add_argument(
        "--cls-is-score",
        action="store_true",
        help="Use this when ONNX outputs raw cls_score without softmax. Recommended for your no-softmax ONNX."
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    detector = RetinaFaceONNX(
        onnx_path=args.onnx,
        input_size=(args.input_size, args.input_size),
        ctx_id=args.ctx_id,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        cls_is_score=args.cls_is_score,
    )

    faces, landmarks = detector.detect(img)

    print("faces shape:", faces.shape)
    print("landmarks shape:", landmarks.shape)

    if faces.shape[0] == 0:
        print("No face detected.")
        return

    print("\nDetected faces:")
    for i in range(faces.shape[0]):
        print(f"[{i}] box+score:", faces[i].tolist())
        print(f"[{i}] landmark5:")
        print(landmarks[i])

    draw_result(img, faces, landmarks, args.save)


if __name__ == "__main__":
    main()