import math
import time

from db.database import DatabaseManager


class PeopleTrackerLogic:
    def __init__(self, max_time: int, radius: int):
        self.max_time = max_time
        self.radius = radius
        # dict: {id: {"pos": (x, y), "start_time": float, "alerted": bool, "last_seen": float}}
        self.track_data = {}

    def process(self, people_detections):
        current_time = time.time()
        active_ids = set()

        for det in people_detections:
            p_id = det["id"]
            if p_id is None:
                continue

            x, y = det["center"]
            active_ids.add(p_id)

            if p_id not in self.track_data:
                self.track_data[p_id] = {
                    "pos": (x, y),
                    "start_time": current_time,
                    "alerted": False,
                    "last_seen": current_time,
                }
            else:
                data = self.track_data[p_id]
                data["last_seen"] = current_time
                old_x, old_y = data["pos"]

                distance = math.sqrt((x - old_x) ** 2 + (y - old_y) ** 2)

                if distance > self.radius:
                    data["pos"] = (x, y)
                    data["start_time"] = current_time
                    data["alerted"] = False
                else:
                    stay_duration = current_time - data["start_time"]
                    if stay_duration > self.max_time and not data["alerted"]:
                        DatabaseManager.push_person_alert(p_id, stay_duration, (x, y))
                        data["alerted"] = True

        to_delete = [
            k for k, v in self.track_data.items() if current_time - v["last_seen"] > 10
        ]
        for k in to_delete:
            del self.track_data[k]
