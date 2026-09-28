"""Live webcam test for the isolated steering processor."""

import time
from urllib.request import urlopen

import cv2
import mediapipe as mp

from steering import SteeringProcessor


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def load_hand_model():
    with urlopen(MODEL_URL, timeout=30) as response:
        model_data = response.read()
    if not model_data:
        raise RuntimeError("The MediaPipe hand model download was empty.")
    return model_data


def run():
    capture = cv2.VideoCapture(CAMERA_INDEX)
    landmarker = None

    try:
        if not capture.isOpened():
            print(f"Cannot open webcam at index {CAMERA_INDEX}.")
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=load_hand_model()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        processor = SteeringProcessor()

        window_name = "Steering Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        previous_timestamp_ms = -1
        previous_frame_time = None
        fps = 0.0

        while True:
            success, frame = capture.read()
            if not success:
                print("Webcam frame read failed; shutting down.")
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = max(time.monotonic_ns() // 1_000_000, previous_timestamp_ms + 1)
            previous_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(image, timestamp_ms)

            wrist_x = None
            wrist_y = None
            hand_detected = bool(result.hand_landmarks)
            if hand_detected:
                landmarks = result.hand_landmarks[0]
                mp.tasks.vision.drawing_utils.draw_landmarks(
                    frame,
                    landmarks,
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )
                wrist = landmarks[0]
                wrist_x = wrist.x * frame_width
                wrist_y = wrist.y * frame_height
                wrist_point = (round(wrist_x), round(wrist_y))
                cv2.circle(frame, wrist_point, 10, (0, 165, 255), -1)
                cv2.circle(frame, wrist_point, 12, (0, 0, 0), 2)

            steering = processor.process(frame_width, wrist_x, hand_detected)
            current_time = time.perf_counter()
            if previous_frame_time is not None:
                elapsed = current_time - previous_frame_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            previous_frame_time = current_time

            hand_status = "DETECTED" if hand_detected else "NOT DETECTED"
            wrist_text = f"{wrist_x:.1f}" if wrist_x is not None else "N/A"
            smoothed_text = (
                f"{processor.smoothed_x * frame_width:.1f}"
                if processor.smoothed_x is not None
                else "N/A"
            )
            info = (
                f"Hand: {hand_status}",
                f"Wrist X: {wrist_text}",
                f"Smoothed X: {smoothed_text}",
                f"Steering: {steering}",
                f"FPS: {fps:.1f}",
            )
            for row, text in enumerate(info):
                cv2.putText(
                    frame,
                    text,
                    (12, 32 + row * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            if landmarker is not None:
                landmarker.close()
        finally:
            try:
                capture.release()
            finally:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    run()