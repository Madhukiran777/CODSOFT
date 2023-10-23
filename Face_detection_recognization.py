# Import libraries
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                         "res10_300x300_ssd_iter_140000.caffemodel")
face_embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Define some constants
FACE_CONFIDENCE = 0.5
FACE_THRESHOLD = 0.4
FACE_SIZE = 96

# Define some variables
known_faces = []
known_names = []
video_capture = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > FACE_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x1, y1, x2, y2 = box.astype("int")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop and resize the face image
            face_image = frame[y1:y2, x1:x2]
            face_image = cv2.resize(face_image, (FACE_SIZE, FACE_SIZE))

            # Convert face image to blob for embedding
            blob = cv2.dnn.blobFromImage(face_image, 1.0 / 255, (FACE_SIZE, FACE_SIZE), (0, 0, 0), swapRB=True)
            face_embedder.setInput(blob)
            embedding = face_embedder.forward()

            # Initialize the name and distance of the recognized face
            name = "Unknown"
            distance = np.inf

            # Loop over the known faces
            for i in range(len(known_faces)):
                similarity = cosine_similarity(embedding, known_faces[i])

                # Check if the similarity is higher than the threshold
                if similarity > FACE_THRESHOLD:
                    name = known_names[i]
                    distance = 1 - similarity

            # Check if the name is unknown
            if name == "Unknown":
                known_faces.append(embedding)
                name = f"Person {len(known_faces)}"
                known_names.append(name)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the name and distance of the face
            cv2.putText(frame, f"{name} ({distance:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and destroy the windows
video_capture.release()
cv2.destroyAllWindows()