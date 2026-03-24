# main.py
import sys

from core.engine import CVEngine


def main():
    print("🚀 Запуск обробки відео...")
    engine = CVEngine()

    total_frames = engine.video.total_frames
    frame_count = 0

    print(f"📁 Всього кадрів для обробки: {total_frames}")

    while True:
        # process_frame тепер повертатиме True (якщо кадр є) або False (якщо кінець)
        has_more_frames = engine.process_frame()

        if not has_more_frames:
            break

        frame_count += 1

        # Виводимо прогрес в один рядок
        sys.stdout.write(
            f"\r⏳ Обробка: {frame_count}/{total_frames} кадрів ({(frame_count / total_frames) * 100:.1f}%)"
        )
        sys.stdout.flush()

    # Закриваємо файли
    engine.stop()
    print("\n✅ Обробка завершена! Відео успішно збережені у папку.")


if __name__ == "__main__":
    main()
