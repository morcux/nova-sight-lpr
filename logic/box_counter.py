import time

import cv2

from db.database import DatabaseManager


class BoxPolygonLogic:
    def __init__(self, polygon, limit: int):
        self.polygon = polygon
        self.limit = limit
        self.last_alert_time = 0
        self.alert_cooldown = 10

    def process(self, boxes_detections):
        inside_count = 0

        for det in boxes_detections:
            x, y = det["center"]
            is_inside = cv2.pointPolygonTest(self.polygon, (x, y), False) >= 0
            if is_inside:
                inside_count += 1

        if inside_count > self.limit:
            current_time = time.time()
            if current_time - self.last_alert_time > self.alert_cooldown:
                DatabaseManager.push_box_alert(inside_count)
                self.last_alert_time = current_time

        return inside_count
