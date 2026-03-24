import numpy as np
import supervision as sv
from roboflow import Roboflow
from roboflow.models.inference import InferenceModel
from ultralytics.models import YOLO


class PeopleTracker:
    def __init__(self, model_path: str, conf: float):
        self.model = YOLO(model_path)
        self.conf = conf

    def track(self, frame):
        results = self.model.track(
            frame, persist=True, conf=self.conf, classes=[0], verbose=False
        )
        return results[0]


class BoxCloudDetector:
    def __init__(self, api_key: str, project_name: str, version: int):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project(project_name)
        self._model = self.project.version(version).model

    @property
    def model(self) -> InferenceModel:
        if not self._model:
            raise ValueError("Model not loaded")
        return self._model

    def detect(self, frame):
        result = self.model.predict(frame, confidence=40).json()
        predictions = result.get("predictions", [])

        if not predictions:
            return sv.Detections.empty()

        xyxy_list = []
        conf_list = []
        class_ids = []

        for p in predictions:
            cx, cy = p["x"], p["y"]
            w, h = p["width"], p["height"]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(p["confidence"] / 100.0)
            class_ids.append(0)

        detections = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.array(conf_list, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )

        return detections
