from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tkinter as tk

# Load the trained model
model = load_model("nlp-pretrain.h5")

# Load the dataset
dataset_path = "./pickle/train/conversation-train.pkl"
with open(dataset_path, "rb") as f:
    dataset = np.load(f, allow_pickle=True)

# Extract questions and answers from the dataset
questions = [item["Questions"] for item in dataset]
answers = [item["Answers"] for item in dataset]

# Fit label encoder with the questions
label_encoder = LabelEncoder()
label_encoder.fit(questions)

# Function to generate a response
def generate_response(input_text):
    if input_text in label_encoder.classes_:
        encoded_input = label_encoder.transform([input_text])
        encoded_input = np.array(encoded_input)
        encoded_input = encoded_input.reshape(1, -1)  # Reshape the input
        predicted_label = np.argmax(model.predict(encoded_input))
        response = label_encoder.inverse_transform([predicted_label])[0]
    else:
        response = "Sorry, I don't have a response for that."
    return response

# Define tkinter GUI
def on_submit():
    user_input = text_input.get("1.0", tk.END).strip()
    
    # Clear the old response
    response_text.config(state=tk.NORMAL)
    response_text.delete("1.0", tk.END)
    
    response = generate_response(user_input)
    
    # Display the new response
    response_text.insert(tk.END, "Bot: " + response)
    response_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Recipe Bot")

text_input = tk.Text(root, height=5, width=30)
text_input.pack()

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

response_text = tk.Text(root, height=5, width=30)
response_text.config(state=tk.DISABLED)
response_text.pack()

root.mainloop()
