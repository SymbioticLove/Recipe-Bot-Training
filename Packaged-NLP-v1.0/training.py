import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, train_questions, train_answers, eval_questions, eval_answers, batch_size, epochs):
    checkpoint = ModelCheckpoint("nlp-pretrain.h5", monitor="val_accuracy", save_best_only=True, mode="max")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)

    train_questions = np.asarray(train_questions)  # Convert to NumPy array if not already
    train_answers = np.asarray(train_answers)  # Convert to NumPy array if not already
    eval_questions = np.asarray(eval_questions)  # Convert to NumPy array if not already
    eval_answers = np.asarray(eval_answers)  # Convert to NumPy array if not already

    model.fit(
        train_questions,
        train_answers,
        validation_data=(eval_questions, eval_answers),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )