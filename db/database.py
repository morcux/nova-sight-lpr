class DatabaseManager:
    @staticmethod
    def push_person_alert(person_id: int, duration: float, position: tuple):
        print(
            f"[DB] ALARM! Людина {person_id} стоїть {int(duration)}с на точці {position}. Запис у БД..."
        )

    @staticmethod
    def push_box_alert(count: int):
        print(f"[DB] ALARM! У зоні забагато коробок ({count} шт). Запис у БД...")
