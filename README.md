# Nova Sight LPR

Nova Sight LPR is a computer vision application built with FastAPI and OpenCV. It is designed for logistics and tracking use cases, such as monitoring worker activity and tracking the number of parcels/boxes in designated zones.

## Features

- **Live Web Stream:** Serves a real-time web interface displaying the processed video stream with annotations.
- **Worker Tracking:** Uses YOLO to track people in the frame. If a worker stays within a certain radius (`STAY_RADIUS_PIXELS`) for longer than a specific time (`STAY_TIME_SECONDS`), an idle alert is triggered and logged to the database.
- **Box Counting in Zones:** Uses the Roboflow API to detect boxes. The system counts the number of boxes inside a predefined polygon zone. If the limit (`BOX_LIMIT`) is exceeded, it triggers an alert.
- **Background Video Recording:** Saves annotated streams of people tracking and box detection to local video files.

## Project Structure

- `main.py`: The entry point for the FastAPI application, serving the video stream and HTML interface.
- `core/`: Contains the CV engine (`engine.py`), video source handler, and object detectors for people (local YOLO) and boxes (Roboflow).
- `logic/`: Implements the business logic for worker tracking (`people_tracker.py`) and box counting (`box_counter.py`).
- `db/`: Basic database manager (`database.py`) to handle alert pushes.
- `config.py`: Configuration management using `pydantic-settings`.

## Prerequisites

- Python 3.10+
- `uv` (or pip) for package management
- A Roboflow account & API Key (for box detection)

## Installation

1. Clone the repository or navigate to the project directory:
   ```bash
   cd nova-sight-lpr
   ```

2. Install dependencies (e.g., using `uv` or `pip` based on the `uv.lock` or `pyproject.toml`):
   ```bash
   uv sync
   # or pip install -r requirements.txt if exported
   ```

3. Ensure you have the required YOLO model (e.g., `yolov8n.pt` or `yolov8l.pt`) in the root directory.

## Configuration

Create a `.env` file in the root directory based on the following environment variables:

```env
# Video Settings
VIDEO_SOURCE="sources/video2.mp4" # Path to video file or 0 for webcam
RESIZE_WIDTH=640

# People tracking (YOLO)
MODEL_PATH="yolov8l.pt"
CONFIDENCE_THRESHOLD=0.3
STAY_TIME_SECONDS=10
STAY_RADIUS_PIXELS=50
PERSON_CLASS_ID=0

# Box detection (Roboflow)
ROBOFLOW_API_KEY="your_roboflow_api_key_here"
ROBOFLOW_PROJECT="parcel-3qci7"
ROBOFLOW_VERSION=1

# Box zone logic
BOX_CLASS_ID=24
BOX_LIMIT=5
POLYGON_POINTS_JSON="[[0, 0], [640, 0],[640, 640], [0, 640]]"
```

## Usage

Start the application using `uvicorn` (or run `main.py` directly):

```bash
python main.py
```
*(By default, the server will run on `http://0.0.0.0:8090`)*

Open your browser and navigate to `http://localhost:8090/` to view the live video stream, "Cam 1: Логістика та Трекінг".
