from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

def build_model(num_labels, max_sequence_length):
    batch_size = 128
    epochs = 50
    # Build the model
    model = Sequential()
    model.add(Embedding(num_labels, 64, input_length=max_sequence_length))
    model.add(LSTM(64, dropout=0.2))
    model.add(Dense(num_labels, activation="softmax"))

    # Add regularization (e.g., dropout) if needed
    # model.add(Dropout(0.2))

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model, batch_size, epochs