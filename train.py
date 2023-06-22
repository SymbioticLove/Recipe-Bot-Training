import random
import pickle
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

# Step 1: Set up your development environment

# Set random seed for reproducibility
random.seed(42)
tf.random.set_seed(42)

# Step 2: Load the data

train_files = ['./pickle/train/chocolate_makers-train.pkl']
eval_files = ['./pickle/eval/chocolate_makers-eval.pkl']

train_labels = []
eval_labels = []
train_cities = []
eval_cities = []
train_states = []
eval_states = []
train_owners = []
eval_owners = []
train_countries = []
eval_countries = []

# Load training data from multiple files
for file_path in train_files:
    with open(file_path, 'rb') as f:
        train_data_list = pickle.load(f)
    for train_data in train_data_list:
        train_labels.append(str(train_data['labels']))
        train_cities.append(str(train_data['City']))
        train_states.append(str(train_data['State']))
        train_owners.append(str(train_data['Owner']))
        train_countries.append(str(train_data['Country']))

# Load evaluation data from multiple files
for file_path in eval_files:
    with open(file_path, 'rb') as f:
        eval_data_list = pickle.load(f)
    for eval_data in eval_data_list:
        eval_labels.append(str(eval_data['labels']))
        eval_cities.append(str(eval_data['City']))
        eval_states.append(str(eval_data['State']))
        eval_owners.append(str(eval_data['Owner']))
        eval_countries.append(str(eval_data['Country']))

# Step 3: Split the data into training and validation sets

train_labels, val_labels, train_cities, val_cities, train_states, val_states, train_owners, val_owners, train_countries, val_countries = train_test_split(
    train_labels, train_cities, train_states, train_owners, train_countries,
    test_size=0.2, random_state=42)


# Step 4: Design your model architecture

# Load pre-trained transformer model and tokenizer
transformer_model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
transformer_model = TFAutoModel.from_pretrained(transformer_model_name)

# Define the recipe bot model
input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
embeddings = transformer_model(input_ids, attention_mask=attention_mask)[0]
pooled_output = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
num_classes = 261
output = tf.keras.layers.Dense(num_classes, activation='softmax')(pooled_output)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Step 5: Train your model

# Define the training parameters (epochs, batch size, etc.)
epochs = 10
batch_size = 32

# Prepare the training and validation datasets as TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_labels, train_cities, train_states, train_owners, train_countries))
train_dataset = train_dataset.shuffle(len(train_labels)).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((val_labels, val_cities, val_states, val_owners, val_countries))
val_dataset = val_dataset.batch(batch_size)

# Define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for step, (batch_y, batch_cities, batch_states, batch_owners, batch_countries) in enumerate(train_dataset):
        text_list = [f"{city}, {state}, {owner}, {country}" for city, state, owner, country in zip(batch_cities.numpy().tolist(), batch_states.numpy().tolist(), batch_owners.numpy().tolist(), batch_countries.numpy().tolist())]

        inputs = tokenizer.batch_encode_plus(
            text_list,  # Pass the concatenated text as a list
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='tf'
        )
        inputs = {k: tf.squeeze(v) for k, v in inputs.items()}
        inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}  # Update input keys
        batch_y = tf.convert_to_tensor(batch_y)

        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss_value = loss_fn(batch_y, logits)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Validation
    val_loss = []
    for batch_y, batch_cities, batch_states, batch_owners, batch_countries in val_dataset:
        text_list = [f"{city}, {state}, {owner}, {country}" for city, state, owner, country in zip(batch_cities.numpy().tolist(), batch_states.numpy().tolist(), batch_owners.numpy().tolist(), batch_countries.numpy().tolist())]

        inputs = tokenizer.batch_encode_plus(
            text_list,  # Pass the concatenated text as a list
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='tf'
        )
        inputs = {k: tf.squeeze(v) for k, v in inputs.items()}
        inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}  # Update input keys
        batch_y = tf.convert_to_tensor(batch_y)

        val_logits = model(inputs, training=False)
        batch_eval_loss = loss_fn(batch_y, val_logits)
        val_loss.append(batch_eval_loss)
    val_loss = tf.reduce_mean(val_loss)
    print(f"Validation Loss: {val_loss}")

# Evaluation loop
eval_loss = []
for batch_y, batch_cities, batch_states, batch_owners, batch_countries in eval_data:
    text_list = [f"{city}, {state}, {owner}, {country}" for city, state, owner, country in zip(batch_cities.numpy().tolist(), batch_states.numpy().tolist(), batch_owners.numpy().tolist(), batch_countries.numpy().tolist())]

    inputs = tokenizer.batch_encode_plus(
        text_list,  # Pass the concatenated text as a list
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='tf'
    )
    inputs = {k: tf.squeeze(v) for k, v in inputs.items()}
    inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}  # Update input keys
    batch_y = tf.convert_to_tensor(batch_y)

    eval_logits = model(inputs, training=False)
    batch_eval_loss = loss_fn(batch_y, eval_logits)
    eval_loss.append(batch_eval_loss)
eval_loss = tf.reduce_mean(eval_loss)
print(f"Evaluation Loss: {eval_loss}")


# Step 7: Save and deploy your model

# Save the model weights and architecture
model.save_weights('model_weights.h5')
model.save('model.h5')
