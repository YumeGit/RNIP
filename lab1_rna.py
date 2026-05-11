# Practical work No. 1
# Topic: Creating an Artificial Neural Network for Binary Classification
# Dataset: Airbnb listings
# Target: superhost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


print("=" * 70)
print("STEP 1-2: Importing libraries and reading the dataset")
print("=" * 70)

data = pd.read_parquet(r"C:\Users\Пользователь\Desktop\RNA\Labs\Lab1\listings.parquet")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nDescriptive statistics:")
print(data.describe())


print("\n" + "=" * 70)
print("STEP 3-7: Exploratory Data Analysis and data cleaning")
print("=" * 70)

print("\nInitial dataset shape:")
print(data.shape)

print("\nMissing values before cleaning:")
print(data.isnull().sum())

columns_to_drop = [
    "listing_id",
    "cover_photo_url",
    "host_id",
    "amenities",
    "currency",
    "country",
    "state",
    "city"
]

data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

data = data.dropna(subset=["superhost"])

data["superhost"] = data["superhost"].astype(str).str.lower()
data["superhost"] = data["superhost"].map({
    "false": 0,
    "true": 1
})

data = data.dropna(subset=["superhost"])
data["superhost"] = data["superhost"].astype(int)

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].fillna("unknown")
    else:
        data[col] = data[col].fillna(data[col].median())

for col in data.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col].astype(str))

print("\nDataset shape after cleaning:")
print(data.shape)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nFirst rows after transformations:")
print(data.head())

print("\nTarget class distribution:")
print(data["superhost"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x=data["superhost"])
plt.title("Target class distribution: superhost")
plt.xlabel("Superhost")
plt.ylabel("Count")
plt.savefig("01_target_class_distribution.png")
plt.close()


print("\n" + "=" * 70)
print("STEP 8-9: Splitting data into training and testing sets")
print("=" * 70)

X = data.drop("superhost", axis=1)
y = data["superhost"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=101,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


print("\n" + "=" * 70)
print("STEP 10-13: Creating, compiling and training the initial model")
print("=" * 70)

model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

print("\nInitial model summary:")
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)


print("\n" + "=" * 70)
print("STEP 14-16: Evaluating the initial model")
print("=" * 70)

predictions = (model.predict(X_test) > 0.5).astype("int32")

print("\nInitial model accuracy:")
print(accuracy_score(y_test, predictions))

print("\nInitial model confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nInitial model classification report:")
print(classification_report(y_test, predictions))

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Initial model - accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("02_initial_model_accuracy.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Initial model - loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("03_initial_model_loss.png")
plt.close()


print("\n" + "=" * 70)
print("STEP 17-18: Adding class weights")
print("=" * 70)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:")
print(class_weights)


print("\n" + "=" * 70)
print("STEP 19-22: Optimizing the artificial neural network")
print("=" * 70)

improved_model = Sequential()
improved_model.add(BatchNormalization(input_shape=(X_train.shape[1],)))
improved_model.add(Dense(128, activation="relu"))
improved_model.add(Dropout(0.3))
improved_model.add(BatchNormalization())
improved_model.add(Dense(64, activation="relu"))
improved_model.add(Dropout(0.3))
improved_model.add(BatchNormalization())
improved_model.add(Dense(32, activation="relu"))
improved_model.add(Dense(1, activation="sigmoid"))

print("\nOptimized model summary:")
improved_model.summary()

improved_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_improved = improved_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)


print("\n" + "=" * 70)
print("STEP 23: Evaluating the optimized model")
print("=" * 70)

improved_predictions = (improved_model.predict(X_test) > 0.5).astype("int32")

print("\nOptimized model accuracy:")
print(accuracy_score(y_test, improved_predictions))

print("\nOptimized model confusion matrix:")
print(confusion_matrix(y_test, improved_predictions))

print("\nOptimized model classification report:")
print(classification_report(y_test, improved_predictions))

plt.figure(figsize=(6, 4))
plt.plot(history_improved.history["accuracy"], label="train")
plt.plot(history_improved.history["val_accuracy"], label="validation")
plt.title("Optimized model - accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("04_optimized_model_accuracy.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(history_improved.history["loss"], label="train")
plt.plot(history_improved.history["val_loss"], label="validation")
plt.title("Optimized model - loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("05_optimized_model_loss.png")
plt.close()


print("\n" + "=" * 70)
print("Execution completed.")
print("Generated graph files:")
print("01_target_class_distribution.png")
print("02_initial_model_accuracy.png")
print("03_initial_model_loss.png")
print("04_optimized_model_accuracy.png")
print("05_optimized_model_loss.png")
print("=" * 70)