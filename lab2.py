# Practical Work No. 2
# Topic: Creating and Applying a Convolutional Neural Network
# Task: Binary image classification - Cats vs Dogs

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalMaxPool2D,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# SETTINGS
# ============================================================
DATASET_PATH = r"C:\Users\Пользователь\Desktop\RNA\Labs\Lab2\dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CNN = 10
EPOCHS_TRANSFER = 5
SEED = 101

# ============================================================
# REMOVE CORRUPTED IMAGES
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp"
)

def clean_bad_images(folder_path):

    removed_files = 0

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            file_path = os.path.join(root, file)

            if not file.lower().endswith(VALID_EXTENSIONS):

                print("Removing non-image file:", file_path)

                os.remove(file_path)

                removed_files += 1

                continue

            try:

                with Image.open(file_path) as img:

                    img = img.convert("RGB")

                    img.save(file_path)

            except Exception:

                print("Removing corrupted image:", file_path)

                os.remove(file_path)

                removed_files += 1

    print("Removed bad files:", removed_files)


clean_bad_images(DATASET_PATH)
print("Dataset cleaned successfully.")

print("=" * 70)
print("STEP 1: Importing libraries and checking dataset folder")
print("=" * 70)

print("Dataset path:", DATASET_PATH)
print("Classes found:", os.listdir(DATASET_PATH))


# ============================================================
# STEP 2-3: Load images and shuffle/split dataset
# ============================================================

print("\n" + "=" * 70)
print("STEP 2-3: Loading images, shuffling and splitting dataset")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

class_names = train_ds.class_names

print("Class names:", class_names)
print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))


# ============================================================
# STEP 4-5: Decode images and get labels
# TensorFlow does this automatically through image_dataset_from_directory
# ============================================================

print("\n" + "=" * 70)
print("STEP 4-5: Images decoded and labels converted to 0 and 1")
print("=" * 70)

print("Label meaning:")
for index, name in enumerate(class_names):
    print(index, "=", name)


# ============================================================
# STEP 6: Create training dataset with performance optimization
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: Preparing training and validation datasets")
print("=" * 70)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ============================================================
# STEP 7: Visualize examples from each class
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: Saving example images from dataset")
print("=" * 70)

plt.figure(figsize=(8, 8))

for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = int(labels[i].numpy()[0])
        plt.title(class_names[label_index])
        plt.axis("off")

plt.savefig("01_dataset_examples.png")
plt.close()

print("Saved: 01_dataset_examples.png")


# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2)
])


# ============================================================
# Helper function for evaluation
# ============================================================

def evaluate_model(model, dataset, model_name):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_classes = (predictions > 0.5).astype("int32")

        y_true.extend(labels.numpy().astype("int32").flatten())
        y_pred.extend(predicted_classes.flatten())

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print(f"Evaluation results for {model_name}")
    print("=" * 70)

    print("Accuracy:", accuracy)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return accuracy


def save_training_plots(history, model_name, file_prefix):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{file_prefix}_accuracy.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{file_prefix}_loss.png")
    plt.close()

    print(f"Saved: {file_prefix}_accuracy.png")
    print(f"Saved: {file_prefix}_loss.png")


# ============================================================
# STEP 8: Create CNN model
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: Creating the CNN model")
print("=" * 70)

cnn_model = Sequential()

cnn_model.add(data_augmentation)
cnn_model.add(Rescaling(1.0 / 255, input_shape=(224, 224, 3)))

# Block 1
cnn_model.add(Conv2D(filters=32, kernel_size=(7, 7), padding="same"))
cnn_model.add(Activation("relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

# Block 2
cnn_model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="valid"))
cnn_model.add(Activation("relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

# Block 3
cnn_model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="valid"))
cnn_model.add(Activation("relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

# Block 4
cnn_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="valid"))
cnn_model.add(Activation("relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="valid"))
cnn_model.add(Activation("relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))

cnn_model.add(GlobalMaxPool2D())
cnn_model.add(Dense(256))
cnn_model.add(Activation("relu"))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(1))
cnn_model.add(Activation("sigmoid"))


# ============================================================
# STEP 9: Show CNN model summary
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: CNN model summary")
print("=" * 70)

cnn_model.summary()


# ============================================================
# STEP 10: Compile and train CNN model
# ============================================================

print("\n" + "=" * 70)
print("STEP 10: Compiling and training CNN model")
print("=" * 70)

cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

cnn_history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_CNN,
    callbacks=[early_stop],
    verbose=1
)

save_training_plots(cnn_history, "CNN", "02_cnn")


# ============================================================
# STEP 11: Evaluate CNN model
# ============================================================

print("\n" + "=" * 70)
print("STEP 11: Evaluating CNN model")
print("=" * 70)

cnn_accuracy = evaluate_model(cnn_model, val_ds, "CNN")


# ============================================================
# TRANSFER LEARNING MODEL FUNCTION
# ============================================================

def create_transfer_model(base_model, model_name):
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)

    if model_name == "ResNet50":
        x = tf.keras.applications.resnet50.preprocess_input(x)
    elif model_name == "InceptionV3":
        x = tf.keras.applications.inception_v3.preprocess_input(x)

    x = base_model(x, training=False)
    x = GlobalMaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# COMPARISON WITH RESNET50
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON MODEL 1: ResNet50")
print("=" * 70)

resnet_base = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

resnet_model = create_transfer_model(resnet_base, "ResNet50")

print("\nResNet50 model summary:")
resnet_model.summary()

resnet_history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_TRANSFER,
    callbacks=[early_stop],
    verbose=1
)

save_training_plots(resnet_history, "ResNet50", "03_resnet50")

resnet_accuracy = evaluate_model(resnet_model, val_ds, "ResNet50")


# ============================================================
# COMPARISON WITH INCEPTIONV3
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON MODEL 2: InceptionV3")
print("=" * 70)

inception_base = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

inception_model = create_transfer_model(inception_base, "InceptionV3")

print("\nInceptionV3 model summary:")
inception_model.summary()

inception_history = inception_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_TRANSFER,
    callbacks=[early_stop],
    verbose=1
)

save_training_plots(inception_history, "InceptionV3", "04_inceptionv3")

inception_accuracy = evaluate_model(inception_model, val_ds, "InceptionV3")


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

results = pd.DataFrame({
    "Model": ["Custom CNN", "ResNet50", "InceptionV3"],
    "Validation Accuracy": [cnn_accuracy, resnet_accuracy, inception_accuracy]
})

print(results)

results.to_csv("05_model_comparison.csv", index=False)

plt.figure(figsize=(7, 4))
plt.bar(results["Model"], results["Validation Accuracy"])
plt.title("Model comparison")
plt.xlabel("Model")
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1)
plt.savefig("05_model_comparison.png")
plt.close()

print("Saved: 05_model_comparison.csv")
print("Saved: 05_model_comparison.png")

print("\nProgram finished successfully.")