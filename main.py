import cv2
import numpy as np
import pyautogui
import time

class GestureController:
    """
    Controls a game using gestures. This version includes a more robust calibration
    that is less sensitive to brightness changes at the edges of the webcam frame.
    """
    def __init__(self):  # <-- Fixed constructor
        pyautogui.FAILSAFE = False  # Optional: avoids issues if mouse hits corner

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cv2.namedWindow("Control Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Control Feed", 960, 540)
        cv2.namedWindow("Color Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color Mask", 640, 360)

        self.left_zone = self.frame_width // 3
        self.right_zone = 2 * self.frame_width // 3
        self.nitro_zone = int(self.frame_height * 0.7)

        self.calibrated = False
        self.is_red = False
        self.hsv_lower1 = None
        self.hsv_upper1 = None
        self.hsv_lower2 = None
        self.hsv_upper2 = None

        print("Gesture Controller Initialized.")
        print("--> Hold your colored object in the center blue square.")
        print("--> Press 'c' to calibrate the color.")
        print("--> Press 'q' to quit.")

    def _release_all_keys(self):
        print("Releasing all keys...")
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        pyautogui.keyUp('shift')

    def calibrate(self, frame):
        calib_area_size = 50
        x_start = (self.frame_width - calib_area_size) // 2
        y_start = (self.frame_height - calib_area_size) // 2

        calib_roi = frame[y_start:y_start + calib_area_size, x_start:x_start + calib_area_size]
        hsv_roi = cv2.cvtColor(calib_roi, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv_roi, axis=(0, 1))
        h = int(mean_hsv[0])

        hue_range = 10

        min_saturation = 50
        min_value = 50

        if h < hue_range or h > 180 - hue_range:
            self.is_red = True
            self.hsv_lower1 = np.array([180 - hue_range, min_saturation, min_value])
            self.hsv_upper1 = np.array([180, 255, 255])
            self.hsv_lower2 = np.array([0, min_saturation, min_value])
            self.hsv_upper2 = np.array([hue_range, 255, 255])
            print("\nRed-like color detected! Using two hue ranges for better tracking.")
        else:
            self.is_red = False
            self.hsv_lower1 = np.array([max(0, h - hue_range), min_saturation, min_value])
            self.hsv_upper1 = np.array([min(180, h + hue_range), 255, 255])
            print(f"\nCalibrated to a non-red color.")

        self.calibrated = True
        print(f"Calibration complete!")
        print("Pressing 'w' for constant forward movement.")
        pyautogui.keyDown('w')
        print("Game controls are now active. Switch to your game window!")
        time.sleep(2)

        print(f"Calibrated HSV hue: {h}")
        print(f"Lower1: {self.hsv_lower1}, Upper1: {self.hsv_upper1}")
        if self.is_red:
          print(f"Lower2: {self.hsv_lower2}, Upper2: {self.hsv_upper2}")


    def _find_object(self, hsv_frame):
        if self.is_red:
            mask1 = cv2.inRange(hsv_frame, self.hsv_lower1, self.hsv_upper1)
            mask2 = cv2.inRange(hsv_frame, self.hsv_lower2, self.hsv_upper2)
            mask = mask1 + mask2
        else:
            mask = cv2.inRange(hsv_frame, self.hsv_lower1, self.hsv_upper1)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 200:
                return largest_contour, mask
        return None, mask

    def _update_controls(self, contour):
        action_text = "Straight"

        if contour is None:
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
            pyautogui.keyUp('shift')
            return "No Object"

        (x, y, w, h) = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        # Debug print (optional)
        print(f"Object Center: ({cx}, {cy}) | Zones: Left < {self.left_zone}, Right > {self.right_zone}, Nitro > {self.nitro_zone}")

        # Determine horizontal movement
        if cx <= self.left_zone:
            pyautogui.keyDown('a')

            pyautogui.keyUp('d')
            action_text = "Left"
        elif cx >= self.right_zone:
            pyautogui.keyDown('d')

            pyautogui.keyUp('a')
            action_text = "Right"
        else:
            pyautogui.keyUp('a')
            pyautogui.keyDown('w')

            pyautogui.keyUp('d')
            action_text = "Straight"

        # Determine Nitro (downward position)
        # Determine Nitro (downward position)
        if cy >= self.nitro_zone:
          # Press space once or twice for Nitro
           pyautogui.press('space')  # or use press('space', presses=2)
           action_text += " + Nitro"


        else:
            pyautogui.keyUp('shift')


        return action_text, (x, y, w, h), (cx, cy)

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                action_text = "CALIBRATE!"
                mask_display = np.zeros_like(frame[:, :, 0])

                if self.calibrated:
                    largest_contour, mask_display = self._find_object(hsv_frame)
                    result = self._update_controls(largest_contour)

                    if isinstance(result, str) and result == "No Object":
                      action_text = "No Object"
                    else:
                      action_text, rect, center = result
                      (x, y, w, h) = rect
                      (cx, cy) = center
                      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                      cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
                else:
                    size = 50
                    x_start = (self.frame_width - size) // 2
                    y_start = (self.frame_height - size) // 2
                    cv2.rectangle(frame, (x_start, y_start), (x_start + size, y_start + size), (255, 0, 0), 2)
                    cv2.putText(frame, "Place color here & press 'c'", (x_start - 50, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw control zones
                cv2.line(frame, (self.left_zone, 0), (self.left_zone, self.frame_height), (255, 255, 0), 2)
                cv2.line(frame, (self.right_zone, 0), (self.right_zone, self.frame_height), (255, 255, 0), 2)
                cv2.line(frame, (0, self.nitro_zone), (self.frame_width, self.nitro_zone), (0, 255, 255), 2)
                cv2.putText(frame, f"Action: {action_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Control Feed", frame)
                cv2.imshow("Color Mask", mask_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and not self.calibrated:
                    self.calibrate(frame)

        finally:
            self._release_all_keys()
            self.cap.release()
            cv2.destroyAllWindows()
            print("Controller shut down.")

if __name__ == "__main__":
    try:
        controller = GestureController()
        controller.run()
    except Exception as e:
        print(f"An error occurred: {e}")
