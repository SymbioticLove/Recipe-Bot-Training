from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def preprocess_data():
    """
    Preprocesses the data from the dataset.

    Args:
        dataset_path: The path to the dataset CSV file.

    Returns:
        padded_sequences: The padded sequences of tokenized questions.
        one_hot_answers: The one-hot encoded answers.
    """
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

    return padded_sequences, one_hot_answers, train_questions, eval_questions, train_answers, eval_answers