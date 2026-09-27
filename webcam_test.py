import cv2

def test_webcam():
    print("🔍 Checking available cameras...")
    # Try different camera indices (0, 1, 2)
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            cap.release()
            break
    else:
        print("❌ No camera found. Try connecting an external webcam or enabling camera access.")
        return

    # Use the working camera
    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    print("🎥 Press 'q' to quit the webcam window.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting webcam test.")
            break

    cap.release()
    cv2.destroyAllWindows()


test_webcam()