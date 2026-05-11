import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Dense,
    Dropout
)
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# SETTINGS
# ============================================================

MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
EPOCHS = 5
BATCH_SIZE = 64

OUTPUT_DIR = "graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("Loading IMDb dataset")
print("=" * 70)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)

x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# ============================================================
# 2. CREATE MODEL FUNCTION
# ============================================================

def create_model(model_type):
    model = Sequential()

    model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN))

    if model_type == "SimpleRNN":
        model.add(SimpleRNN(64))
    elif model_type == "LSTM":
        model.add(LSTM(64))
    elif model_type == "GRU":
        model.add(GRU(64))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 3. TRAIN AND EVALUATE MODELS
# ============================================================

models = ["SimpleRNN", "LSTM", "GRU"]
results = []

histories = {}

for model_name in models:
    print("\n" + "=" * 70)
    print("Training model:", model_name)
    print("=" * 70)

    model = create_model(model_name)

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"{model_name} Test Loss:", test_loss)
    print(f"{model_name} Test Accuracy:", test_accuracy)

    results.append({
        "Model": model_name,
        "Loss": test_loss,
        "Accuracy": test_accuracy
    })

    histories[model_name] = history

    if model_name == "GRU":
        final_model = model


# ============================================================
# 4. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("Model comparison")
print("=" * 70)
print(results_df)


plt.figure(figsize=(7, 4))

plt.bar(results_df["Model"], results_df["Accuracy"])

plt.title("RNN Models Comparison")

plt.xlabel("Model")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_models_comparison.png"
    )
)

plt.close()


# ============================================================
# 5. TRAINING GRAPHS
# ============================================================

for model_name, history in histories.items():

    # Accuracy graph

    plt.figure(figsize=(6, 4))

    plt.plot(history.history["accuracy"], label="train")

    plt.plot(
        history.history["val_accuracy"],
        label="validation"
    )

    plt.title(f"{model_name} Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_accuracy.png"
        )
    )

    plt.close()

    # Loss graph

    plt.figure(figsize=(6, 4))

    plt.plot(history.history["loss"], label="train")

    plt.plot(
        history.history["val_loss"],
        label="validation"
    )

    plt.title(f"{model_name} Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_loss.png"
        )
    )

    plt.close()


# ============================================================
# 6. REAL-TIME TESTING WITH NEW VALUES
# ============================================================

word_index = imdb.get_word_index()

def encode_review(text):
    words = text.lower().split()
    encoded = []

    for word in words:
        index = word_index.get(word, 2)
        if index < MAX_WORDS:
            encoded.append(index + 3)
        else:
            encoded.append(2)

    return pad_sequences([encoded], maxlen=MAX_LEN)


def predict_review(text):
    encoded_text = encode_review(text)

    prediction = final_model.predict(encoded_text, verbose=0)[0][0]

    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    print("\nText:", text)
    print("Prediction:", sentiment)
    print("Confidence:", round(float(prediction) * 100, 2), "%")


predict_review("This movie was amazing and very interesting")
predict_review("This film was boring and terrible")
predict_review("The actors were good but the story was bad")