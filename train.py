import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Add detaset path ####

train_dir = "dataset/train"
test_dir = "dataset/test"


## Prepare the Image ##

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

## Load the training Images ##

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    subset="training"
)

## Load the validation images ##

validation_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    subset="validation"
)

## load test images ##

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

## Create the CNN model ##

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(7, activation="softmax")
])

## Input layer shape ##

layers.Input(shape=(48, 48, 1)),

## First convolution ##

layers.Conv2D(32, (3, 3), activation="relu"),

## First pooling layer ##

layers.MaxPooling2D((2, 2)),

## Second convolution ##

layers.Conv2D(64, (3, 3), activation="relu"),

## Second pooling layer ##

layers.MaxPooling2D((2, 2)),

## Third convolution ##

layers.Conv2D(128, (3, 3), activation="relu"),

## Third pooling layer ##

layers.MaxPooling2D((2, 2)),

## Flatten the output ##

layers.Flatten(),

## Dense layer ##

layers.Dense(128, activation="relu"),

## Dropout layer ##

layers.Dropout(0.5),

## Final layer ##

layers.Dense(7, activation="softmax")

## compile the model ##

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

## Display the model summary ##

model.summary()

## Train the model ##

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20
)

model.save("model/emotion_model.keras")

print("Model saved successfully!")

import matplotlib.pyplot as plt

# Accuracy graph
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Loss graph
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
