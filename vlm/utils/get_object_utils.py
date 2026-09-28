import os
from collections import Counter

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


def remove_combined_mode(cfg):
    """Whether GroundingDINO's label decoder respects caption segment boundaries.

    False is ApexNav's shipped behaviour and stays the default, so configs
    written before this ablation are unaffected. See GroundingDINO.predict()'s
    docstring for what the flag actually changes -- in short, False decodes every
    caption token above text_threshold into one string, merging several category
    names together, while True bounds the decode to the segment containing the
    box's argmax token.
    """
    mode = getattr(cfg.groundingDINO, "remove_combined", False)
    if not isinstance(mode, bool):
        raise ValueError(
            f"detector.groundingDINO.remove_combined must be a bool, got {mode!r}"
        )
    return mode


def detector_threshold(cfg):
    """The configured score gate, on the ACTIVE backend's scale.

    The two are not interchangeable -- the detectors do not score on a common
    scale, so reading the wrong key measures "tuned vs untuned" rather than one
    detector against the other.

    The often-quoted "0.081 vs 0.256, so 0.40 is OWLv2's p99" figures were
    measured on frames where the target was mostly ABSENT and describe
    false-alarm scores only; on target-present frames OWLv2 reaches 0.814.
    See analysis/detector/FINDINGS.md's 2026-09-28 banner.
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
        remove_combined=remove_combined_mode(cfg),
    )


DEFAULT_LABEL_MATCH = "exact"


def label_match_mode(cfg):
    """'exact' (the shipped behaviour) or 'fuzzy'. Defaults to exact, so configs
    written before this ablation keep working untouched."""
    mode = getattr(cfg.groundingDINO, "label_match", DEFAULT_LABEL_MATCH)
    if mode not in ("exact", "fuzzy"):
        raise ValueError(
            f"detector.groundingDINO.label_match must be 'exact' or 'fuzzy', got {mode!r}"
        )
    return mode


def load_label_vocabulary(llm_answer_path):
    """Every category name the dataset knows about -- the keys of the llm_answer
    file. Used as DISTRACTORS by label_match(); see there for why.

    Returns [] rather than raising if the file is missing, because the two
    non-sweep entry points (habitat_manual_control.py, real_world_test) may run
    without one and must keep working.
    """
    vocab = []
    try:
        with open(llm_answer_path) as fh:
            for line in fh:
                if line.strip().startswith("#") or ":" not in line:
                    continue
                vocab.append(line.split(":", 1)[0].strip())
    except OSError:
        return []
    # Pipe-separated goals ("table | dining table | ...") are several names.
    out = []
    for name in vocab:
        out.extend(n.strip() for n in name.split("|"))
    return out


def label_match(phrase, candidates, mode=DEFAULT_LABEL_MATCH, distractors=()):
    """Which of `candidates` a detector's returned `phrase` should count as.

    Returns the matched candidate string, or None. `candidates` MUST be ordered
    with the goal's own labels first (i.e. right_label_list + similar), because
    that order is the tie-break -- see below.

    `distractors` are OTHER category names the dataset knows about, which compete
    for the match but can never win it: if one of them matches MORE specifically
    than any candidate, the phrase belongs to something that is not in play here
    and None is returned. Without this the rule is asymmetric in a way that
    manufactures false positives -- see "THE DISTRACTOR GUARD" below.

    WHY THIS EXISTS
    ---------------
    The two detectors do not return the same KIND of thing, and ApexNav's
    original check assumed they did.

    OWLv2 is handed a LIST of queries and returns, per box, an INDEX into it. Its
    label is therefore always one of the exact strings we asked for, and an
    exact-string filter over it is a no-op (805/805 survived on ovon ep 61).

    GroundingDINO is handed ONE concatenated caption
    ('sink cabinet.  cabinet.  bath cabinet.  kitchen cabinet.  drawer.  ') and
    performs PHRASE GROUNDING: per box it reports which caption TOKENS fired,
    glued back together. That is a readout of the caption, not a category name,
    so it routinely returns strings that are in no candidate list at all. Real
    output from ep 61, target 'sink cabinet', 117 detections:

         61  kitchen cabinet
         21  cabinet cabinet cabinet kitchen cabinet
         16  cabinet kitchen cabinet
          4  sink kitchen cabinet
          2  sink cabinet cabinet cabinet kitchen cabinet

    Under exact matching NONE of those counted as the target -- 0 of 117 -- even
    though several plainly contain it. Measured at 27% of detections lost across
    captions, 48% on this one.

    So the exact filter does not merely favour OWLv2, it conflates two different
    things: how well a detector SEES, and whether its output format happens to
    fit ApexNav's plumbing. 'fuzzy' exists to separate them, as a third arm --
    NOT as a fix to the baseline. 'exact' must stay byte-identical, or the
    baseline moves under us and every banked run becomes incomparable.

    THE RULE: token-set containment, most specific wins.
      * lowercase token multisets, so repeated tokens are not free
      * a candidate matches when its tokens are contained in the phrase's
      * among matches, the one with the MOST tokens wins

    "Most specific wins" is load-bearing. 'cabinet kitchen cabinet' contains
    'cabinet' but NOT 'sink cabinet', so it is a similar. But
    'sink cabinet cabinet cabinet kitchen cabinet' contains both, and assigning
    it to the generic 'cabinet' (a similar, slot != 0) rather than to the target
    would defeat the entire point of the arm.

    Ties on token count are broken by ORDER IN `candidates`, which is why the
    goal's own labels must come first: 'sink kitchen cabinet' contains both
    'sink cabinet' and 'kitchen cabinet' at two tokens each, and the target has
    to win. Without that the arm's headline case is decided by dict ordering.

    THE DISTRACTOR GUARD
    --------------------
    Containment is ASYMMETRIC, and OVON's vocabulary is full of compound names
    that share a head noun. val_seen alone has five separate cabinet categories
    -- `cabinet`, `bathroom cabinet`, `sink cabinet`, `display cabinet`,
    `kitchen lower cabinet` -- each with its own goal instances.

      target `bathroom cabinet`, phrase 'cabinet'          -> no match. SAFE:
          'bathroom' is missing, so containment fails.
      target `cabinet`, phrase 'bathroom cabinet'          -> MATCHES, and that
          is wrong: a bathroom cabinet is a DIFFERENT annotated object, so the
          agent would drive to the wrong thing and fail the episode.

    So generic targets absorb their specific siblings. Measured on val_seen:
    17 targets have a more-specific sibling and 9 mis-awarded it. The other 8
    were safe only because the LLM happened to list the sibling as a similar --
    luck, not design.

    Passing the dataset's category vocabulary as `distractors` makes that a
    guarantee: 'bathroom cabinet' matches the distractor at two tokens, beating
    the candidate `cabinet` at one, so the detection is dropped rather than
    credited to the target. Distractors are checked AFTER candidates with a
    strict >, so an exact tie always goes to the candidate.

    THE RISK, WHICH IS REAL: relaxing this raises false positives, and
    C: False Positive is already 33 of 104 agent failures (32%). A looser match
    can promote EXPLORE to SEARCH_OBJECT on the wrong object and burn the
    500-step budget. This arm may well score WORSE. That is a result, not a
    failure, and it gets reported either way.
    """
    if mode == "exact":
        # Identity. Kept inside this function rather than branched at the call
        # site so both arms go through exactly one code path.
        return phrase if phrase in candidates else None

    phrase_tokens = Counter(phrase.lower().split())
    best, best_n, best_wins = None, 0, False
    candidate_set = set(candidates)
    for is_candidate, pool in ((True, candidates), (False, distractors)):
        for name in pool:
            # A distractor that is already a candidate must not block it.
            if not is_candidate and name in candidate_set:
                continue
            tokens = Counter(name.lower().split())
            if not tokens or tokens - phrase_tokens:   # not contained
                continue
            n = sum(tokens.values())
            if n > best_n:                   # strict >, so ties keep the earlier
                best, best_n = name, n       # entry -- candidates before
                best_wins = is_candidate     # distractors, goal before similars
    return best if best_wins else None


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

def get_object(right_label, img, cfg, similar_answer, distractors=()):
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
        mode = label_match_mode(cfg)
        # Resolved ONCE against the union, then asked which list won -- not two
        # independent membership tests. A phrase can contain BOTH the target and
        # a similar ('sink cabinet cabinet cabinet kitchen cabinet' contains
        # 'sink cabinet' and 'cabinet'), and testing sequentially would award it
        # to whichever list happens to be checked first. right_label_list is
        # placed first so it also wins ties on specificity.
        candidates = right_label_list + [l for l in dino_label if l not in right_label_list]
        for idx in range(len(detections.logits)):
            label_detected = label_match(
                detections.phrases[idx], candidates, mode, distractors)
            score = detections.logits[idx].item()
            if label_detected is None:
                continue
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

