from data_preprocessing import preprocess_data
from model_building import build_model
from training import train_model
from saving import save_model
from evaluation import evaluate_model

# Preprocess the data
padded_sequences, one_hot_answers, train_questions, eval_questions, train_answers, eval_answers = preprocess_data()

# Build the model
model, epochs, batch_size = build_model(one_hot_answers.shape[1], padded_sequences.shape[1])

# Train the model
batch_size = train_model(model, train_questions, train_answers, eval_questions, eval_answers, epochs, batch_size)

# Evaluate the model
evaluate_model(model, eval_questions, eval_answers, batch_size)

# Save the trained model
save_model(model, "nlp-pretrain.h5")