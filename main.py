from contextlib import asynccontextmanager

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from core.engine import CVEngine

engine = CVEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if engine.people_writer:
        engine.people_writer.release()
    if engine.boxes_writer:
        engine.boxes_writer.release()
    engine.video.stop()


app = FastAPI(title="CV Engine WebStream", lifespan=lifespan)


def generate_frames():
    while True:
        frame = engine.process_frame()
        if frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CV Engine - Live Stream</title>
        <style>
            body { margin: 0; padding: 0; background-color: #121212; color: #ffffff; font-family: 'Segoe UI', Tahoma, sans-serif; display: flex; flex-direction: column; align-items: center; height: 100vh; }
            h1 { margin-top: 2rem; font-size: 2rem; color: #00ff88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.3); }
            .video-container { margin-top: 2rem; border: 2px solid #333; border-radius: 10px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); background-color: #000; }
            img { display: block; max-width: 100%; height: auto; }
            .status { margin-top: 1rem; color: #888; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <h1>Cam 1: Логістика та Трекінг</h1>
        <div class="video-container">
            <img src="/video_feed" alt="Live Video Stream">
        </div>
        <div class="status">🟢 Stream is live | Обробка записується у фоні</div>
    </body>
    </html>
    """
    return html_content


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090, log_level="info")
