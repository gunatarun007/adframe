import cv2
cap = cv2.VideoCapture('/workspace/demo.mp4')
ret, frame = cap.read()
if ret:
    cv2.imwrite('/workspace/adframe/frame0.jpg', frame)
    print("Frame 0 saved successfully!")
else:
    print("Failed to read frame 0.")
cap.release()
