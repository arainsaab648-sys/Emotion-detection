import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# Dataset paths
# =========================

train_dir = "dataset/train"
test_dir = "dataset/test"

# =========================
# Training data
# =========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# =========================
# Test data
# =========================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# =========================
# Load training images
# =========================

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    subset="training"
)

# =========================
# Validation data
# =========================

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

validation_data = validation_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# =========================
# Load test images
# =========================

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# =========================
# CNN Model
# =========================

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(7, activation="softmax")
])

# =========================
# Compile
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Train ONCE
# =========================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=30
)

# =========================
# Training results
# =========================

print("\nTraining Results:")

for i in range(len(history.history["accuracy"])):
    print(
        f"Epoch {i+1}: "
        f"Training Accuracy = {history.history['accuracy'][i]:.4f}, "
        f"Validation Accuracy = {history.history['val_accuracy'][i]:.4f}"
    )

# =========================
# Evaluate test dataset
# =========================

test_loss, test_accuracy = model.evaluate(test_data)

print("\n================================")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("================================")

# =========================
# Save model
# =========================

model.save("model/emotion_model_final.keras")

print("New model saved successfully!")