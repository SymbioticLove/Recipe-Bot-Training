Hello! Thank you for dowloading. This is an early release, v1.0. Open setup.py in any text editor to see the dependency requirements for your virtual environment!

Known issues: Lack of datasets, will need to modify components to handle multiple datasets, one-hot encoding is far from ideal for NLP - looking into a Python library that is supposed to be better.
<div>&nbsp</div>
CURRENTLY, THE MODEL ARCHITECHTURE, CHECKPOINT AND WEIGHTS ALL SAVE APPROPRIATELY IN THE "PRETRAINED-MODEL" DIRECTORY. THERE IS ALSO AN NLP-PRETRAIN.H5 FILE THAT IS SAVED IN THE ROOT DIRECTORY, AND I CANNOT FOR THE LIFE OF ME FIGURE OUT WHY THAT IS. ANYONE WHO SOLVES THIS WILL BE MY BEST BUDDY FOR AT LEAST A WEEK.
<div>&nbsp</div>
Make sure you have Python downloaded from https://www.python.org/downloads/
<div>&nbsp</div>
The .bat files included in the "shortcuts" directory can be used to open the command shell and execute the various actions. You should execute "Create_Env" first as this will create a virtual envrionment and install the necessary dependencies. If you need to test the dependences, run "Dep_Test". "Run" executes the actual training script. These shortcuts all assume that you have downloaded the root directory, "Packaged-NLP-v1.0", at your C: drive. If you have it elsewhere, you will need to modify these. This is a very early release, and the current single dataset is highly unlikely to produce any relevant conversation or even a result at all. This is also due to the encoding. If you have any suggestions, reach out to matthew@symbiotic.love.
<div>&nbsp</div>
The master.py script imports 5 individual components (preprocess_data, build_model, train_model, evaluate_model, save_model) and operates a script in the order given to produce the checkpoint version of the trained bot, as well as it's architecture and weights to be built upon further. The full.py script located in the unpackaged directory is the same script functionally, but is not componetized in any way. I highly advise using the packaged and componetized version, as this is easier to modify. In detail, the script functions as follows:
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
  <div>&nbsp</div>
  <li>Loads the .csv dataset
  <div>&nbsp</div>

  ```python
  dataset_path = "./datasets/Conversation.csv"
  dataset = pd.read_csv(dataset_path)
  ```
    
  </li>
  <div>&nbsp</div>
  <li>Creates a dictionary to map old column names to new column names
  <div>&nbsp</div>

  ```python
  column_mapping = {
    "question": "Questions",
    "answer": "Answers"
  }
  ```

  </li>
  <div>&nbsp</div>
  <li>Renames the columns of the directory using the column mapping directory
  <div>&nbsp</div>

  ```python
  dataset = dataset.rename(columns=column_mapping)
  ```
  
  </li>
  <div>&nbsp</div>
  <li>Extracts questions and answers from the dataset
  <div>&nbsp</div>

  ```python
  questions = dataset["Questions"].values
  answers = dataset["Answers"].values
  ```

  </li>
  <div>&nbsp</div>
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
  <div>&nbsp</div>
  <li>Tokenizes and converts the questions to sequences
  <div>&nbsp</div>

  ```python
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(questions)
  sequences = tokenizer.texts_to_sequences(questions)
  ```
    
  </li>
  <div>&nbsp</div>
  <li>Determines the maximum sequence length and pads the sequences to have a consistent length
  <div>&nbsp<div>
    
  ```python
  max_sequence_length = max(len(seq) for seq in sequences)
  padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
  ```
    
  </li>
  <div>&nbsp</div>
  <li>Splits the dataset into training and evaluation data
  <div>&nbsp<div>
    
  ```python
  train_questions, eval_questions, train_answers, eval_answers = train_test_split(
    padded_sequences, one_hot_answers, test_size=0.2, random_state=42
  )
  ```
    
  </li>
  <div>&nbsp</div>
  <li>Builds the model
  <div>&nbsp<div>
    
  ```python
  model = Sequential()
  model.add(Embedding(num_labels, 64, input_length=max_sequence_length))
  model.add(LSTM(64, dropout=0.2))
  model.add(Dense(num_labels, activation="softmax"))
  ```
    
  </li>
  <div>&nbsp</div>
  <li>Compiles the model with the loss function, optimizer, and metrics
  <div>&nbsp<div>
    
  ```python
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  ```
    
  </li>
  <div>&nbsp</div>
  <li>Defines the callbacks for model checkpointing and early stopping
  <div>&nbsp<div>
    
  ```python
  checkpoint = ModelCheckpoint("nlp-pretrain.h5", monitor="val_accuracy", save_best_only=True, mode="max")
  early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
  ```
    
  </li>
  <div>&nbsp</div>
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
  <div>&nbsp</div>
  <li>Saves the trained model
  <div>&nbsp<div>
    
  ```python
  model.save("nlp-pretrain.h5")
  ```
    
  </li>
  <div>&nbsp</div>
  <li>Prints the prediction accuracy of the model
  <div>&nbsp<div>
    
  ```python
  _, accuracy = model.evaluate(eval_questions, eval_answers, batch_size=batch_size)
  print("Prediction accuracy:", accuracy)
  ```
    
  </li>
</ol>

PLAIN TEXT (line numbers from full.py)

1. Imports the necessary libraries and modules (lines 1-9)
2. Loads the .csv dataset (lines 12-13)
3. Creates a dictionary to map old column names to new column names (lines 16-19)
4. Renames the columns of the directory using the column mapping directory (line 22)
5. Extracts questions and answers from the dataset(s) (lines 25-26)
6. Converts the answers to one-hot encoding - non-ideal (29-33)
7. Tokenizes and convert the questions to sequences (lines 36-38)
8. Determines the maximum sequence length and pads the sequences to be consistent (lines 41-44)
9. Splits the dataset into training (80%) and evaluation (20%) data (lines 47-48)
10. Builds the model (lines 52-55)
11. Compiles the model with the loss function, optimizer, and metrics (line 61)
12. Defines the callbacks for model checkpointing and early stopping (lines 64-65)
13. Trains the model (lines 68-77)
14. Saves the trained model (line 80)
15. Prints the prediction accuracy of the model (lines 83-84)
