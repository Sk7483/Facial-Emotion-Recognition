
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------
# Load trained model
# -------------------------
print("Loading model...")
model = load_model("emotion_model.h5")
print("Model loaded successfully")

# -------------------------
# Emotion labels
# -------------------------
emotion_labels = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']

# -------------------------
# Load Haar Cascade
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# Start Webcam
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opening")
    exit()

print("Camera started")

# -------------------------
# Real-time Detection Loop
# -------------------------

emotion_buffer = []
buffer_size = 10   # bigger = more stable but slower

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi, verbose=0)
emotion_buffer.append(np.argmax(prediction))

if len(emotion_buffer) > buffer_size:
    emotion_buffer.pop(0)

# Get most common emotion in buffer
emotion = emotion_labels[max(set(emotion_buffer), key=emotion_buffer.count)]


        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()



