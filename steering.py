"""Convert a detected wrist position into a stable steering state."""


class SteeringProcessor:
    """Smooth normalized wrist positions and classify them as steering states.

    The left and right thresholds bound the center dead zone. Hysteresis
    extends the current side's boundary to prevent state chatter.
    """

    def __init__(
        self,
        smoothing_factor=0.35,
        left_threshold=0.40,
        right_threshold=0.60,
        hysteresis=0.04,
    ):
        if not 0 < smoothing_factor <= 1:
            raise ValueError("smoothing_factor must be greater than 0 and at most 1")
        if not 0 <= left_threshold < 0.5 < right_threshold <= 1:
            raise ValueError("thresholds must surround 0.5 and stay within [0, 1]")
        if hysteresis < 0 or 2 * hysteresis >= right_threshold - left_threshold:
            raise ValueError("hysteresis must be non-negative and narrower than the dead zone")

        self.smoothing_factor = smoothing_factor
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.hysteresis = hysteresis
        self.smoothed_x = None
        self.state = "NEUTRAL"

    def process(self, frame_width, wrist_x, hand_detected):
        """Return LEFT, CENTER, RIGHT, or NEUTRAL for the current frame."""
        if not hand_detected:
            self.reset()
            return self.state

        if frame_width <= 0:
            raise ValueError("frame_width must be positive")
        if wrist_x is None:
            raise ValueError("wrist_x is required when a hand is detected")

        normalized_x = min(max(wrist_x / frame_width, 0.0), 1.0)
        if self.smoothed_x is None:
            self.smoothed_x = normalized_x
        else:
            alpha = self.smoothing_factor
            self.smoothed_x = alpha * normalized_x + (1 - alpha) * self.smoothed_x

        self.state = self._classify(self.smoothed_x)
        return self.state

    def reset(self):
        """Clear smoothing and steering state, for example after hand loss."""
        self.smoothed_x = None
        self.state = "NEUTRAL"

    def _classify(self, position):
        if self.state == "LEFT":
            if position < self.left_threshold + self.hysteresis:
                return "LEFT"
        elif self.state == "RIGHT":
            if position > self.right_threshold - self.hysteresis:
                return "RIGHT"

        if position < self.left_threshold:
            return "LEFT"
        if position > self.right_threshold:
            return "RIGHT"
        return "CENTER"