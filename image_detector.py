import cv2
import numpy as np
import tensorflow as tf

# =========================
# Load trained model
# =========================

model = tf.keras.models.load_model(
    "model/emotion_model.keras"
)

# Emotion labels
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# =========================
# Load face detector
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================
# Ask for image path
# =========================

image_path = input("Enter image path: ")

# Read image
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not open image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

if len(faces) == 0:
    print("No face detected.")
    exit()

print("Face detected!")

# =========================
# Detect emotion
# =========================

for (x, y, w, h) in faces:

    # Crop face
    face = gray[y:y+h, x:x+w]

    # Resize to model input size
    face = cv2.resize(face, (48, 48))

    # Normalize
    face = face.astype("float32") / 255.0

    # Add dimensions
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    # Predict
    prediction = model.predict(face, verbose=0)

    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    confidence = np.max(prediction) * 100

    print("Emotion:", emotion)
    print("Confidence:", f"{confidence:.2f}%")

    # Draw face rectangle
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Display emotion
    cv2.putText(
        image,
        f"{emotion} {confidence:.1f}%",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# =========================
# Show result
# =========================

cv2.imshow("Emotion Detection", image)

print("Press Q to close the result window.")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()