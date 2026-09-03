import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the trained model
model = tf.keras.models.load_model(
    "model/emotion_model.keras"
)


# Test dataset path
test_dir = "dataset/test"


# Prepare test images
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# Load test dataset
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)


print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)