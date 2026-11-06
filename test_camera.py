
import cv2

print("OpenCV version:", cv2.__version__)

for index in [0, 1, 2]:
    print(f"\nTesting camera {index}...")

    cap = cv2.VideoCapture(
        index,
        cv2.CAP_AVFOUNDATION
    )

    print("Opened:", cap.isOpened())

    if cap.isOpened():
        ret, frame = cap.read()
        print("Read frame:", ret)

        if ret:
            print("Frame shape:", frame.shape)

            cv2.imshow("Camera Test", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

    cap.release()

print("\nTesting finished.")
