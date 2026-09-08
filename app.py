import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="😊",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    text-align: center;
    margin-top: 20px;
}

.emotion {
    font-size: 38px;
    font-weight: 700;
}

.confidence {
    font-size: 22px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="main-title">😊 AI Emotion Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a screenshot or image and let the CNN model detect facial emotion'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/emotion_model.keras"
    )


model = load_model()


# ==========================================
# EMOTION LABELS
# ==========================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ==========================================
# LOAD FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload a screenshot or image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Convert PIL image to NumPy
    image_array = np.array(image)

    # Convert RGB to OpenCV BGR
    frame = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # ======================================
    # DISPLAY ORIGINAL IMAGE
    # ======================================

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        width="stretch"
    )

    st.divider()

    # ======================================
    # CHECK FACE
    # ======================================

    if len(faces) == 0:

        st.warning(
            "⚠️ No face detected. "
            "Please upload a clear image containing a face."
        )

    else:

        st.success(
            f"✅ {len(faces)} face(s) detected!"
        )

        result_image = frame.copy()

        # ==================================
        # PROCESS EACH FACE
        # ==================================

        for face_number, (x, y, w, h) in enumerate(faces, start=1):

            # Crop face
            face = gray[
                y:y+h,
                x:x+w
            ]

            # Resize to 48 × 48
            face = cv2.resize(
                face,
                (48, 48)
            )

            # Normalize
            face = face.astype(
                "float32"
            ) / 255.0

            # Add batch dimension
            face = np.expand_dims(
                face,
                axis=0
            )

            # Add channel dimension
            face = np.expand_dims(
                face,
                axis=-1
            )

            # ==================================
            # PREDICT EMOTION
            # ==================================

            prediction = model.predict(
                face,
                verbose=0
            )

            emotion_index = np.argmax(
                prediction
            )

            emotion = emotion_labels[
                emotion_index
            ]

            confidence = (
                np.max(prediction) * 100
            )

            # ==================================
            # DRAW FACE BOX
            # ==================================

            cv2.rectangle(
                result_image,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                result_image,
                f"{emotion} {confidence:.1f}%",
                (x, max(y-10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ==================================
            # RESULT
            # ==================================

            st.markdown(
                f"""
                <div class="result-box">

                <div>Face {face_number}</div>

                <div class="emotion">
                😊 {emotion}
                </div>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # ======================================
        # SHOW DETECTED IMAGE
        # ======================================

        st.subheader("🔍 Detection Result")

        result_image_rgb = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_image_rgb,
            width="stretch"
        )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("ℹ️ About")

    st.write(
        """
        **AI Emotion Detector**

        This application uses a CNN
        trained on facial emotion images.

        ### Supported emotions

        😠 Angry

        🤢 Disgust

        😨 Fear

        😊 Happy

        😐 Neutral

        😢 Sad

        😲 Surprise

        ### Input

        Upload a screenshot or image
        containing one or more faces.

        ### Model

        `emotion_model.keras`

        ### Processing

        Face Detection → Grayscale →
        48×48 → CNN → Emotion
        """
    )