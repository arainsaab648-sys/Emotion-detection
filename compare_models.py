import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# Models to compare
models_to_test = [
    "model/emotion_model.keras",
    "model/emotion_model_new.keras",
    "model/emotion_model_final.keras"
]

print("\n================================")
print("MODEL COMPARISON")
print("================================")

for model_path in models_to_test:

    print("\nTesting:", model_path)

    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(test_data, verbose=1)

    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    print("Accuracy Percentage:", accuracy * 100, "%")

print("\n================================")
print("Comparison finished!")
print("================================")