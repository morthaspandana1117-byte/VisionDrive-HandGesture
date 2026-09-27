import time
from urllib.request import urlopen

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
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


def get_hand_region(x_pixel, frame_width):
    if x_pixel < frame_width / 3:
        return "LEFT"
    if x_pixel < 2 * frame_width / 3:
        return "CENTER"
    return "RIGHT"


def run():
    capture = cv2.VideoCapture(CAMERA_INDEX)
    landmarker = None

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        model_data = load_hand_model()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        cv2.namedWindow("Hand Tracking Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Tracking Test", 960, 720)

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

            # Use the mirrored display frame so measured left/right matches what the user sees.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = max(time.monotonic_ns() // 1_000_000, previous_timestamp_ms + 1)
            previous_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(image, timestamp_ms)

            left_boundary = frame_width // 3
            right_boundary = 2 * frame_width // 3
            cv2.line(frame, (left_boundary, 0), (left_boundary, frame_height), (255, 200, 0), 2)
            cv2.line(frame, (right_boundary, 0), (right_boundary, frame_height), (255, 200, 0), 2)
            cv2.putText(frame, "LEFT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(frame, "CENTER", (left_boundary + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(frame, "RIGHT", (right_boundary + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            wrist_x = None
            wrist_y = None
            region = "N/A"
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                mp.tasks.vision.drawing_utils.draw_landmarks(
                    frame,
                    landmarks,
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )

                # The wrist (landmark 0) is a simple, stable first point to inspect.
                wrist = landmarks[0]
                wrist_x = wrist.x * frame_width
                wrist_y = wrist.y * frame_height
                region = get_hand_region(wrist_x, frame_width)
                wrist_point = (round(wrist_x), round(wrist_y))
                cv2.circle(frame, wrist_point, 12, (0, 165, 255), -1)
                cv2.circle(frame, wrist_point, 14, (0, 0, 0), 2)

            current_time = time.perf_counter()
            if previous_frame_time is not None:
                elapsed = current_time - previous_frame_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            previous_frame_time = current_time

            hand_status = "DETECTED" if wrist_x is not None else "NOT DETECTED"
            x_text = f"{wrist_x:.1f}" if wrist_x is not None else "N/A"
            y_text = f"{wrist_y:.1f}" if wrist_y is not None else "N/A"
            info = (
                f"Hand: {hand_status}",
                f"X: {x_text}",
                f"Y: {y_text}",
                f"Region: {region}",
                f"FPS: {fps:.1f}",
            )
            for row, text in enumerate(info):
                cv2.putText(
                    frame,
                    text,
                    (12, 62 + row * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Hand Tracking Test", frame)
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