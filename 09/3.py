import cv2

# Load the Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(1) # Use 0 for the default camera

# Loop to read video frames continuously
while True:
    ret, frame = cap.read()
    if not ret:
        break # Exit the loop if frame reading fails

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Corrected COLOR_2BGRGRAY to COLOR_BGR2GRAY

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle, thickness 2

    # Display the result in a window
    cv2.imshow('Deteksi Wajah Realtime', frame) # Corrected "REaltime" to "Realtime"

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()