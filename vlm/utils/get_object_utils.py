import os

import cv2
import numpy as np
from vlm.coco_classes import COCO_CLASSES
from vlm.detector.yolov7 import YOLOv7Client
from vlm.segmentor.sam import MobileSAMClient
from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.detector.owlv2 import OWLv2Client
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.utils.get_itm_message import get_itm_message

yolov7_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", 12184)))
blip2_itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2_PORT", 12182)))
sam_segmentor = MobileSAMClient(port=int(os.environ.get("SAM_PORT", 12183)))
# Open-vocabulary detector clients. Both are just URL holders -- constructing one
# does not touch a model or a GPU -- so building both up front costs nothing and
# keeps the backend choice a pure lookup at call time.
dino_detector = GroundingDINOClient(port=int(os.environ.get("GDINO_PORT", 12181)))
owlv2_detector = OWLv2Client(port=int(os.environ.get("OWLV2_PORT", 12185)))

DEFAULT_DETECTOR_BACKEND = "groundingDINO"


def detector_backend(cfg):
    """Which open-vocabulary detector this run uses.

    Defaults to groundingDINO when `detector.backend` is absent, so configs
    written before the OWLv2 ablation keep working untouched.
    """
    backend = getattr(cfg, "backend", DEFAULT_DETECTOR_BACKEND)
    if backend not in ("groundingDINO", "owlv2"):
        raise ValueError(
            f"detector.backend must be 'groundingDINO' or 'owlv2', got {backend!r}"
        )
    return backend


def detector_threshold(cfg):
    """The configured score gate, on the ACTIVE backend's scale.

    The two are not interchangeable: OWLv2's median detection scores 0.081
    against GroundingDINO's 0.256, so GroundingDINO's shipped 0.40 sits at
    roughly OWLv2's 99th percentile. Reading the wrong key measures
    "tuned vs untuned" rather than one detector against the other.
    See analysis/detector/FINDINGS.md.
    """
    if detector_backend(cfg) == "owlv2":
        return cfg.owlv2.confidence_threshold_owlv2
    return cfg.groundingDINO.confidence_threshold_dino


def llm_answer_path_for(path, cfg):
    """The llm_answer file matching the ACTIVE detector, by convention
    `<stem>_owlv2<ext>` for OWLv2.

    Why this has to switch with the backend: each llm_answer line ends with the
    paper's per-category `c_th`, the gate `isConfidenceObject` tests the fused
    confidence against. That fused value is a convex combination of raw detector
    scores (paper Eq. 2), so it never leaves the detector's own scale -- and
    neither does c_th. GroundingDINO's c_th runs up to 0.65 while OWLv2's whole
    score distribution tops out near 0.43, so reusing the GroundingDINO file
    would leave 161 of ovon's 177 categories with a gate they can essentially
    never clear. That is the "tuned vs untuned" comparison the calibration in
    analysis/detector/ exists to prevent.

    Raises rather than falling back, because a silent fallback is precisely the
    failure this guards against. hm3dv1/hm3dv2 have no variant and do not need
    one: all six HM3D-ObjectNav goals are COCO classes, so slot 0 is fed by
    YOLOv7 and the swap does not touch the target path there. If you hit this
    error on hm3d, the fix is to use ovon or mp3d, not to generate a file.
    """
    if detector_backend(cfg) != "owlv2":
        return path
    stem, ext = os.path.splitext(path)
    if stem.endswith("_owlv2"):
        return path  # already named explicitly by an override
    owlv2_path = f"{stem}_owlv2{ext}"
    if not os.path.exists(owlv2_path):
        raise FileNotFoundError(
            f"detector.backend=owlv2 needs OWLv2-scale c_th values, but "
            f"{owlv2_path} does not exist. Generate it with "
            f"analysis/detector/rescale_cth.py, or the run would apply "
            f"GroundingDINO-scale thresholds to OWLv2 scores."
        )
    return owlv2_path


def detect(cfg, img, caption):
    """Run the configured detector. The ONLY place the backend is branched on.

    Both clients return ObjectDetections with normalized xyxy boxes and exact
    label strings in `.phrases`, so every consumer below is backend-agnostic.
    """
    if detector_backend(cfg) == "owlv2":
        # text_threshold has no OWLv2 analogue -- it gates GroundingDINO's
        # phrase-grounding step, which OWLv2 does not have -- so it is not passed.
        return owlv2_detector.predict(
            img, caption=caption, box_threshold=cfg.owlv2.confidence_threshold_owlv2
        )
    return dino_detector.predict(
        img,
        caption=caption,
        box_threshold=cfg.groundingDINO.confidence_threshold_dino,
        text_threshold=cfg.groundingDINO.text_threshold,
    )


def get_segmentation(segmented_img, idx, detections, img, label, score, color):
    object_mask = np.zeros((480, 640), dtype=np.uint8)
    bbox_denorm = detections.boxes[idx] * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]

    if bbox_area / img_area < 0.99:
        object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(segmented_img, [contour], 0, color, 4)

        cv2.rectangle(
            segmented_img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
        )
        label_x = x1
        label_y = y1 - text_height
        cv2.rectangle(
            segmented_img,
            (label_x, label_y - 30),
            (label_x + text_width, label_y + text_height),
            color,
            2,
        )
        cv2.putText(
            segmented_img,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )

    return segmented_img, object_mask

def get_object(right_label, img, cfg, similar_answer):
    score_list = []
    object_masks_list = []
    segmented_img = img.copy()
    label_list = []
    coco_label = []
    dino_label = []
    right_label_list = list(map(str.strip, right_label.split('|')))
    # print(f"right_label_list: {right_label_list}")
    all_answer = right_label_list + similar_answer
    for label in all_answer:
        if label in COCO_CLASSES:
            coco_label.append(label)
        else:
            dino_label.append(label)

    if any(item in dino_label for item in right_label_list):
        dino_label = all_answer
        coco_label = []
        for label in right_label_list:
            if label in COCO_CLASSES:
                coco_label.append(label)

    if coco_label:
        detections = yolov7_detector.predict(img, agnostic_nms=cfg.yolo.agnostic_nms, 
                                            conf_thres=cfg.yolo.confidence_threshold_yolo, iou_thres=cfg.yolo.iou_threshold_yolo)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] in right_label_list:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)
            elif detections.phrases[idx] in coco_label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(list(all_answer).index(label_detected) - len(right_label_list)+1)

    if dino_label:
        caption = ' '.join(f'{item}.  ' for item in dino_label)
        detections = detect(cfg, img, caption)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if label_detected in right_label_list:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)

            elif label_detected in dino_label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(list(all_answer).index(label_detected) - len(right_label_list)+1)

    return segmented_img, score_list, object_masks_list, label_list

def get_object_with_itm(label, img, cfg):
    score_list = []
    object_masks_list = []
    cosine_list = []
    itm_score_list = []
    segmented_img = img.copy()
    if label in COCO_CLASSES:
        detections = yolov7_detector.predict(img, agnostic_nms=cfg.yolo.agnostic_nms,
                                             conf_thres=cfg.yolo.confidence_threshold_yolo, iou_thres=cfg.yolo.iou_threshold_yolo)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] == label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine:.3f}, itm_score: {itm_score:.3f}")
                score_list.append(score)
                object_masks_list.append(object_mask)
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)

    else:
        detections = detect(cfg, img, label)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if score > detector_threshold(cfg):
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine}, itm_score: {itm_score}")
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)
    
    return segmented_img, score_list, object_masks_list, cosine_list, itm_score_list


def crop_and_expand_box(img, detections, idx, expand_pixels=0.4):
    # Get bounding box coordinates in [x_min, y_min, x_max, y_max] format
    x_min, y_min, x_max, y_max = detections.boxes[idx]
    x_min = int(x_min * img.shape[1])
    y_min = int(y_min * img.shape[0])
    x_max = int(x_max * img.shape[1])
    y_max = int(y_max * img.shape[0])

    # Expand the box outward; clamp to image boundaries
    x_min = max(int(x_min*(1-expand_pixels)), 0)
    y_min = max(int(y_min*(1-expand_pixels)), 0)
    x_max = min(int(x_max*(1+expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max*(1+expand_pixels)), img.shape[0] - 1)

    # Crop the image to keep only the box region
    img_detected = img[y_min:y_max+1, x_min:x_max+1]

    return img_detected

