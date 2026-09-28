"""Map stable steering states to stateful A/D key events."""

import pyautogui


class KeyboardInputMapper:
	"""Hold only the steering key required by the current steering state."""

	_VALID_STATES = {"LEFT", "CENTER", "RIGHT", "NEUTRAL"}

	def __init__(self, keyboard=None):
		self._keyboard = keyboard if keyboard is not None else pyautogui
		self._held_keys = set()

	@property
	def keyboard_state(self):
		"""Return the held key for display, or NONE when neither is held."""
		if "a" in self._held_keys:
			return "A"
		if "d" in self._held_keys:
			return "D"
		return "NONE"

	def update(self, steering_state):
		"""Apply a steering state without repeating keyDown/keyUp events."""
		if steering_state not in self._VALID_STATES:
			raise ValueError(f"Unsupported steering state: {steering_state!r}")

		if steering_state == "LEFT":
			self._release("d")
			self._press("a")
		elif steering_state == "RIGHT":
			self._release("a")
			self._press("d")
		else:
			self.release_all()

		return self.keyboard_state

	def release_all(self):
		"""Release A and D if held; safe to call repeatedly."""
		first_error = None
		for key in ("a", "d"):
			try:
				self._release(key)
			except Exception as error:
				if first_error is None:
					first_error = error
		if first_error is not None:
			raise first_error

	def _press(self, key):
		if key not in self._held_keys:
			self._keyboard.keyDown(key)
			self._held_keys.add(key)

	def _release(self, key):
		if key in self._held_keys:
			self._keyboard.keyUp(key)
			self._held_keys.remove(key)
