<h1>Ultimate Kitchen Companion AI</h1>
Here you will find all of the scripts that I am using to train the ultimate kitchen companion in the form of an AI chatbot. This bot will be able to suggest recipes based on myriad factors such as taste, ethnicity, type, caloric content and more! Think, "I'd like a recipe for an Indian dish that isn't too spicy and has under 400 calories."
<h2>6/25 Update</h2>
Leaps and bounds have been made this week. The scripts are now combined into a single script capable of training and AI using .csv datasets (though, this could be modified to use any type of file) and a NLP model. The checkpoint and early_stopping callbacks ensure that only the best model from testing is saved and that the training is stopped if there is no improvement for 5 epochs, preventing overfitting and diminishing returns. The finalized files are saved as an .h5 file called "nlp-pretrain" so it can be interacted with or built upon further. The repository files now include the env files, a requirements .txt, and both a full.py file that contains the entire script without componetization, as well as a master.py file and all of the components packaged for download. The script functions as follows:
<div>&nbsp</div>
<ol>
  <li>Imports the necessary libraries and modules
  <div>&nbsp</div>
  
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
  ```

  </li>
  <li>Loads the .csv dataset
  <div>&nbsp</div>

  ```python
  dataset_path = "./datasets/Conversation.csv"
  dataset = pd.read_csv(dataset_path)
  ```
    
  </li>
  <li>Creates a dictionary to map old column names to new column names
  <div>&nbsp</div>

  ```python
  column_mapping = {
    "question": "Questions",
    "answer": "Answers"
  }
  ```

  </li>
  <li>Renames the columns of the directory using the column mapping directory
  <div>&nbsp</div>

  ```python
  dataset = dataset.rename(columns=column_mapping)
  ```
  
  </li>
  <li>Extracts questions and answers from the dataset
  <div>&nbsp</div>

  ```python
  questions = dataset["Questions"].values
  answers = dataset["Answers"].values
  ```

  </li>
  <li>Converts the answers to one-hot encoding (non-ideal, this is being worked on)
  <div>&nbsp</div>
    
  ```python
  label_to_index = {label: index for index, label in enumerate(set(answers))}
  index_to_label = {index: label for label, index in label_to_index.items()}
  encoded_answers = np.array([label_to_index[label] for label in answers])
  num_labels = len(label_to_index)
  one_hot_answers = to_categorical(encoded_answers, num_labels)
  ```
    
  </li>
  <li>Tokenizes and converts the questions to sequences
  <div>&nbsp</div>

  ```python
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(questions)
  sequences = tokenizer.texts_to_sequences(questions)
  ```
    
  </li>
  <li>Determines the maximum sequence length and pads the sequences to have a consistent length
  <div>&nbsp<div>
    
  ```python
  max_sequence_length = max(len(seq) for seq in sequences)
  padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
  ```
    
  </li>
  <li>Splits the dataset into training and evaluation data
  <div>&nbsp<div>
    
  ```python
  train_questions, eval_questions, train_answers, eval_answers = train_test_split(
    padded_sequences, one_hot_answers, test_size=0.2, random_state=42
  )
  ```
    
  </li>
  <li>Builds the model
  <div>&nbsp<div>
    
  ```python
  model = Sequential()
  model.add(Embedding(num_labels, 64, input_length=max_sequence_length))
  model.add(LSTM(64, dropout=0.2))
  model.add(Dense(num_labels, activation="softmax"))
  ```
    
  </li>
  <li>Compiles the model with the loss function, optimizer, and metrics
  <div>&nbsp<div>
    
  ```python
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  ```
    
  </li>
  <li>Defines the callbacks for model checkpointing and early stopping
  <div>&nbsp<div>
    
  ```python
  checkpoint = ModelCheckpoint("nlp-pretrain.h5", monitor="val_accuracy", save_best_only=True, mode="max")
  early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
  ```
    
  </li>
  <li>Trains the model
  <div>&nbsp<div>
    
  ```python
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

  ```
    
  </li>
  <li>Saves the trained model
  <div>&nbsp<div>
    
  ```python
  model.save("nlp-pretrain.h5")
  ```
    
  </li>
  <li>Prints the prediction accuracy of the model
  <div>&nbsp<div>
    
  ```python
  _, accuracy = model.evaluate(eval_questions, eval_answers, batch_size=batch_size)
  print("Prediction accuracy:", accuracy)
  ```
    
  </li>
</ol>
The one-hot encoding is not ideal for NLP processing. I am looking into better conversational approaches currently.

