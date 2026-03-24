import json

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    VIDEO_SOURCE: str
    RESIZE_WIDTH: int = 640
    MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5

    ROBOFLOW_API_KEY: str
    ROBOFLOW_PROJECT: str
    ROBOFLOW_VERSION: int

    STAY_TIME_SECONDS: int = 60
    STAY_RADIUS_PIXELS: int = 50
    BOX_LIMIT: int = 5
    POLYGON_POINTS_JSON: str

    OUTPUT_PEOPLE_VIDEO: str = "output_people.mp4"
    OUTPUT_BOXES_VIDEO: str = "output_boxes.mp4"
    OUTPUT_FPS: int = 20

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def POLYGON_POINTS(self) -> np.ndarray:
        points_list = json.loads(self.POLYGON_POINTS_JSON)
        return np.array(points_list, np.int32)


config = Settings()  # type: ignore
