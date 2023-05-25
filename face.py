import cv2

# Load a sample image and learn how to recognize it
known_image = cv2.imread("known_face.jpg")
known_encoding = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Compare each detected face to the known face
    for (x, y, w, h) in faces:
        # Extract the face region from the color frame
        face_image = frame[y:y+h, x:x+w]
        face_encoding = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Compare the face encoding with the known face encoding
        match = cv2.compareHist(known_encoding, face_encoding, cv2.HISTCMP_CORREL)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Write the name of the recognized person
        if match > 0.7:
            name = "Known Person"
        else:
            name = "Unknown"
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()
