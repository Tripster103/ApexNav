"""OWLv2 open-vocabulary detector, drop-in replacement for GroundingDINO.

Same three-layer shape as grounding_dino.py -- model class, client stub, and a
__main__ that hosts the model on a port -- so nothing downstream has to change.

The inference core is lifted from analysis/detector/owlv2_smoke_test.py, which
was validated against GroundingDINO on byte-identical inputs on 2026-08-25
(see analysis/detector/FINDINGS.md). Two things from that run carry over:

  * Box coordinates. OWLv2 pads the input to a SQUARE with grey before resizing,
    padding bottom/right only, so raw boxes are normalized against max(H,W).
    transformers 4.43.2 already corrects for this inside
    post_process_object_detection (image_processing_owlv2.py:528 does
    `size = torch.max(img_h, img_w)`), so passing the true (H, W) is correct and
    needs no workaround. What it does NOT handle is a detection landing in the
    grey pad region below y=H; those are dropped here.

  * Score scale. OWLv2's scores are NOT on GroundingDINO's scale. Passing 0.40
    here would admit almost nothing and would measure "tuned vs untuned".

    CORRECTED 2026-09-28: the "median 0.081 against 0.256" and the 0.25 matched
    point both come from hm3dv1 frames where the target was mostly ABSENT, so
    they describe FALSE-ALARM scores. On target-present frames OWLv2 scores much
    higher (fitted c_th up to 0.814) and the volume-matched point is 0.320, not
    0.25. See analysis/detector/FINDINGS.md's banner.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""

import os
from typing import List, Optional

import numpy as np
import torch

from vlm.detector.detections import ObjectDetections

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

# Guarded exactly as grounding_dino.py and yolov7.py guard theirs: this module is
# imported by the planner for its *client*, on processes that have no model and
# no GPU. A hard import here would break those.
try:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
except ImportError:
    print("Could not import OWLv2. This is OK if you are only using the client.")

MODEL_NAME = os.environ.get("OWLV2_MODEL", "google/owlv2-base-patch16-ensemble")


def caption_to_queries(caption: str) -> List[str]:
    """ApexNav's caption convention -> OWLv2's query list.

    get_object_utils.py builds captions for GroundingDINO as
    'chair.  sofa.  bed.  ' -- period-separated, because that is what phrase
    grounding wants. OWLv2 instead scores a LIST of independent text queries, so
    the same string is split back apart here. Doing the conversion inside the
    server keeps the client signature identical to GroundingDINOClient's, which
    is what makes this a drop-in.
    """
    return [q.strip() for q in caption.split(".") if q.strip()]


class OWLv2:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        )

    def predict(
        self,
        image: np.ndarray,
        caption: Optional[str] = "",
        box_threshold: Optional[float] = 0.25,
    ) -> ObjectDetections:
        """Detect `caption`'s categories in `image`.

        Returns ObjectDetections with boxes as NORMALIZED xyxy -- the same
        convention GroundingDINO's output lands in, and what get_segmentation()
        expects when it multiplies back up by image width/height.
        """
        h, w = image.shape[:2]
        queries = caption_to_queries(caption)
        if not queries:
            return self._empty(image)

        # flush=True: stdout is block-buffered to a log file, and the server is
        # SIGTERMed at run end, so unflushed lines are lost outright -- which is
        # why the 2026-09-15 smoke test could not show what was detected.
        print("OWLv2 is detecting. Caption:", caption, flush=True)
        inputs = self.processor(
            text=[queries], images=image, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        res = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=box_threshold,
            target_sizes=torch.tensor([[h, w]], device=self.device),
        )[0]

        boxes, logits, phrases = [], [], []
        for score, label, box in zip(
            res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()
        ):
            # Centred in the grey pad region -> not a real detection. The pad is
            # on the bottom AND right, so which axis overruns depends on the
            # sensor's orientation: a LoCoBot frame (640x480 landscape) is padded
            # below, but OVON's Stretch sensor is 360x640 PORTRAIT and is padded
            # to the right instead. Checking only y would let every detection in
            # the right-hand grey strip through on ovon, where it would then be
            # clamped to a degenerate sliver at x=1.0 rather than dropped.
            # owlv2_smoke_test.py only ever ran landscape frames, which is why it
            # needed the y check alone.
            if 0.5 * (box[1] + box[3]) > h or 0.5 * (box[0] + box[2]) > w:
                continue
            # Absolute pixels -> normalized, to match GroundingDINO's convention.
            #
            # Clamped to [0, 1] because post_process_object_detection scales by
            # max(H, W), not by (W, H) -- so a box overlapping the grey pad comes
            # back larger than the real image: up to 640/480 = 1.33 on a LoCoBot
            # frame, and 640/360 = 1.78 in x on OVON's portrait sensor.
            # GroundingDINO's boxes are always within [0, 1], and two consumers
            # rely on that: get_segmentation() (get_object_utils.py:19) multiplies
            # straight back up and would hand SAM an out-of-frame box, and
            # annotate() (detections.py:161) uses `box.max() <= 1` to decide
            # whether to de-normalize at all, so a >1 box is silently drawn wrong.
            boxes.append(
                [
                    min(max(box[0] / w, 0.0), 1.0),
                    min(max(box[1] / h, 0.0), 1.0),
                    min(max(box[2] / w, 0.0), 1.0),
                    min(max(box[3] / h, 0.0), 1.0),
                ]
            )
            logits.append(score)
            # `label` is an INDEX into `queries`, not a string. Mapping it back
            # here is what lets the two call sites in get_object_utils.py keep
            # doing `detections.phrases[idx] in right_label_list` unchanged.
            phrases.append(queries[label])

        # The caption print above says what was ASKED for; this says what came
        # BACK. Without it a log cannot distinguish "detector ran and found
        # nothing" from "detector ran and found plenty" -- both are HTTP 200.
        print(
            f"OWLv2 found {len(boxes)} detection(s)"
            + (": " + ", ".join(f"{p}={l:.3f}" for p, l in zip(phrases, logits))
               if boxes else ""),
            flush=True,
        )

        if not boxes:
            return self._empty(image)

        return ObjectDetections(
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(logits, dtype=torch.float32),
            phrases,
            image_source=image,
            fmt="xyxy",  # already xyxy; do NOT let it convert from cxcywh
        )

    @staticmethod
    def _empty(image: np.ndarray) -> ObjectDetections:
        return ObjectDetections(
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            [],
            image_source=image,
            fmt="xyxy",
        )


class OWLv2Client:
    def __init__(self, port: int = int(os.environ.get("OWLV2_PORT", 12185))):
        self.url = f"http://localhost:{port}/owlv2"

    def predict(
        self,
        image_numpy: np.ndarray,
        caption: Optional[str] = "",
        box_threshold: Optional[float] = 0.25,
        text_threshold: Optional[float] = None,
    ) -> ObjectDetections:
        """`text_threshold` is accepted and IGNORED.

        It exists only so the two call sites in get_object_utils.py can pass the
        same keyword arguments to either backend. OWLv2 has no phrase-grounding
        step for it to gate -- it scores query embeddings against boxes directly
        -- so the parameter is meaningless rather than transferable, and is not
        sent over the wire.
        """
        response = send_request(
            self.url,
            image=image_numpy,
            caption=caption,
            box_threshold=box_threshold,
        )
        return ObjectDetections.from_json(response, image_source=image_numpy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12185)
    args = parser.parse_args()

    print("Loading model...")

    class OWLv2Server(ServerMixin, OWLv2):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(
                image,
                caption=payload["caption"],
                box_threshold=payload["box_threshold"],
            ).to_json()

    owlv2 = OWLv2Server()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(owlv2, name="owlv2", port=args.port)
