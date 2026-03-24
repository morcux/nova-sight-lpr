import time

import cv2
import supervision as sv

from config import config
from core.detector import BoxCloudDetector, PeopleTracker
from core.video_source import VideoSource
from logic.box_counter import BoxPolygonLogic
from logic.people_tracker import PeopleTrackerLogic


class CVEngine:
    def __init__(self):
        self.video = VideoSource(config.VIDEO_SOURCE)

        self.people_tracker = PeopleTracker(
            config.MODEL_PATH, config.CONFIDENCE_THRESHOLD
        )
        self.box_detector = BoxCloudDetector(
            config.ROBOFLOW_API_KEY, config.ROBOFLOW_PROJECT, config.ROBOFLOW_VERSION
        )

        self.people_logic = PeopleTrackerLogic(
            config.STAY_TIME_SECONDS, config.STAY_RADIUS_PIXELS
        )
        self.box_logic = BoxPolygonLogic(config.POLYGON_POINTS, config.BOX_LIMIT)

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.last_api_call = 0
        self.api_cooldown = 0.5
        self.last_box_detections = None

        self.people_writer = None
        self.boxes_writer = None

    def process_frame(self):
        ret, frame = self.video.get_frame()
        if not ret or frame is None:
            return None

        frame = cv2.resize(
            frame,
            (
                config.RESIZE_WIDTH,
                int(frame.shape[0] * (config.RESIZE_WIDTH / frame.shape[1])),
            ),
        )

        if self.people_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

            original_fps = self.video.fps

            self.people_writer = cv2.VideoWriter(
                config.OUTPUT_PEOPLE_VIDEO, fourcc, original_fps, (w, h)
            )
            self.boxes_writer = cv2.VideoWriter(
                config.OUTPUT_BOXES_VIDEO, fourcc, original_fps, (w, h)
            )

        frame_people = frame.copy()
        frame_boxes = frame.copy()

        people_dets = []
        box_dets = []

        people_results = self.people_tracker.track(frame)
        if people_results.boxes is not None and people_results.boxes.id is not None:
            boxes = people_results.boxes.xyxy.cpu().numpy()  # type: ignore
            track_ids = people_results.boxes.id.int().cpu().numpy()  # type: ignore

            for box, t_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                people_dets.append({"id": t_id, "center": (cx, cy)})

                for f in [frame_people, frame]:
                    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        f,
                        f"Worker ID: {t_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        current_time = time.time()
        if current_time - self.last_api_call > self.api_cooldown:
            self.last_box_detections = self.box_detector.detect(frame)
            self.last_api_call = current_time

        if self.last_box_detections is not None:
            for xyxy in self.last_box_detections.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                box_dets.append({"center": (cx, cy)})

                cv2.circle(frame_boxes, (cx, cy), 4, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            frame_boxes = self.box_annotator.annotate(
                scene=frame_boxes, detections=self.last_box_detections
            )
            frame = self.box_annotator.annotate(
                scene=frame, detections=self.last_box_detections
            )

        self.people_logic.process(people_dets)
        boxes_in_zone = self.box_logic.process(box_dets)

        for f in [frame_boxes, frame]:
            cv2.polylines(
                f,
                [config.POLYGON_POINTS],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.putText(
                f,
                f"Boxes in Zone: {boxes_in_zone}/{config.BOX_LIMIT}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        self.people_writer.write(frame_people)
        if self.boxes_writer:
            self.boxes_writer.write(frame_boxes)

        return frame
