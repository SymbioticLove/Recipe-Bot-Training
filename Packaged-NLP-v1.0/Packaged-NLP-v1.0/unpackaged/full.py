import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load the dataset from CSV
dataset_path = "./datasets/Conversation.csv"
dataset = pd.read_csv(dataset_path)

# Create a dictionary to map old column names to new column names
column_mapping = {
    "question": "Questions",
    "answer": "Answers"
}

# Rename the columns using the column_mapping dictionary
dataset = dataset.rename(columns=column_mapping)

# Extract questions and answers
questions = dataset["Questions"].values
answers = dataset["Answers"].values

# Convert answers to one-hot encoding
label_to_index = {label: index for index, label in enumerate(set(answers))}
index_to_label = {index: label for label, index in label_to_index.items()}
encoded_answers = np.array([label_to_index[label] for label in answers])
num_labels = len(label_to_index)
one_hot_answers = to_categorical(encoded_answers, num_labels)

# Tokenize and convert questions to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)

# Determine the maximum sequence length
max_sequence_length = max(len(seq) for seq in sequences)

# Pad sequences to have consistent length
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Split the dataset into training and evaluation data
train_questions, eval_questions, train_answers, eval_answers = train_test_split(
    padded_sequences, one_hot_answers, test_size=0.2, random_state=42
)

# Build the model
model = Sequential()
model.add(Embedding(num_labels, 64, input_length=max_sequence_length))
model.add(LSTM(64, dropout=0.2))
model.add(Dense(num_labels, activation="softmax"))

# Add regularization (e.g., dropout) if needed
# model.add(Dropout(0.2))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define callbacks
checkpoint = ModelCheckpoint("nlp-pretrain.h5", monitor="val_accuracy", save_best_only=True, mode="max")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)

# Train the model
batch_size = 128
epochs = 50
model.fit(
    train_questions,
    train_answers,
    validation_data=(eval_questions, eval_answers),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping]
)

# Save the trained model
model.save("nlp-pretrain.h5")

# Print prediction accuracy
_, accuracy = model.evaluate(eval_questions, eval_answers, batch_size=batch_size)
print("Prediction accuracy:", accuracy)
