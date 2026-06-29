import cv2

# Open the default webcam (0 = first camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Webcam started")
print("Press 'q' to quit")

while True:
    # Read one frame from the webcam
    success, frame = cap.read()

    if not success:
        print("❌ Failed to read frame")
        break

    # Display the frame
    cv2.imshow("SignBridge Camera Test", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()