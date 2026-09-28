import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
import groundingdino.datasets.transforms as T
from vlm.detector.detections import ObjectDetections

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from groundingdino.util.inference import load_model, predict
except ModuleNotFoundError:
    print(
        "Could not import groundingdino. This is OK if you are only using the client."
    )

GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


class GroundingDINO:
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(
            model_config_path=config_path, model_checkpoint_path=weights_path
        ).to(device)
        self.caption = caption

    def predict(
        self,
        image: np.ndarray,
        caption: Optional[str] = None,
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
        remove_combined: Optional[bool] = False,
    ) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.
            remove_combined (Optional[bool]): how the returned LABEL is decoded
                from the model's per-token scores. False is upstream's default
                and ApexNav's shipped behaviour; True is the ablation arm.

                The model never returns text. It returns pred_logits of shape
                (n_boxes, 256 token positions) -- a score per box per CAPTION
                TOKEN. The label string is manufactured afterwards by
                get_phrases_from_posmap (util/utils.py:599).

                  False: take EVERY token above text_threshold, anywhere in the
                         caption, and decode them together (inference.py:90-95).
                         Our captions are category lists, so tokens from several
                         different categories merge into one string. Real output
                         for target 'sink cabinet':
                         'cabinet cabinet cabinet kitchen cabinet' -- which is
                         not any category name, so get_object()'s exact-match
                         filter discards it. 27.4% of detections lost this way.

                  True:  find the box's ARGMAX token, use the separator ids
                         101/1012/102 ([CLS] / '.' / [SEP]) to bound decoding to
                         the ONE caption segment containing it (inference.py:81-89).
                         One segment = one category name.

                The model itself already separates the categories -- the paper's
                "sub-sentence level" text representation uses block-diagonal
                attention masks explicitly to block attention between unrelated
                category names (arXiv:2303.05499 Sec. 3.1). It is only the
                DECODER that ignores those boundaries. False is the right default
                for referring expressions ("the man in the red shirt"), where the
                answer really is a multi-word span; it is the wrong one for a
                category list. Inherited from VLFM, which does the same.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        image_tensor = F.to_tensor(image)
        image_transformed = F.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        # flush=True and a result line, mirroring vlm/detector/owlv2.py exactly.
        # Without these the two arms are not comparable from logs: stdout is
        # block-buffered to a file and the server is SIGTERMed at run end, so
        # unflushed lines are lost. Added 2026-09-15 to measure whether the
        # target-vs-similar label competition differs between the detectors.
        print("GroundingDINO is detecting. Caption:", caption_to_use,
              f"| remove_combined={remove_combined}", flush=True)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                remove_combined=remove_combined,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)
        print(
            f"GroundingDINO found {len(phrases)} detection(s)"
            + (": " + ", ".join(f"{p}={l:.3f}" for p, l in zip(phrases, logits.tolist()))
               if len(phrases) else ""),
            flush=True,
        )

        # Remove detections whose class names do not exactly match the provided classes
        # classes = caption_to_use[: -len(" .")].split(" . ")
        # detections.filter_by_class(classes)

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = int(os.environ.get("GDINO_PORT", 12181))):
        self.url = f"http://localhost:{port}/gdino"

    def predict(
        self,
        image_numpy: np.ndarray,
        caption: Optional[str] = "",
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
        remove_combined: Optional[bool] = False,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            remove_combined=remove_combined,
        )
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(
                image,
                caption=payload["caption"],
                box_threshold=payload["box_threshold"],
                text_threshold=payload["text_threshold"],
                # .get, not [...]: a server started before this key existed must
                # not 500 on an old client, and vice versa. Absent == shipped
                # behaviour, which is the safe default.
                remove_combined=payload.get("remove_combined", False),
            ).to_json()

    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
