import pandas as pd
import pickle

# Load the dataset from a CSV file
df = pd.read_csv('./[PATH TO YOUR DATASET].csv')

# Extract the labels from a unique column
labels = df['UNIQUE COLUMN NAME']

# Clean and normalize the data (if required)
# No specific cleaning or normalization steps needed for this dataset

# Rename the columns
df = df.rename(columns={
    'labels': 'labels',
    'COLUMN NAME FROM DATASET': 'NEW COLUMN NAME',
    'COLUMN NAME FROM DATASET': 'NEW COLUMN NAME'
})

# Split the dataset into training and evaluation sets (70% for training, 30% for evaluation)
train_df = df.sample(frac=0.7, random_state=1)
eval_df = df.drop(train_df.index)

# Convert the DataFrames to dictionaries
train_data = train_df.to_dict(orient='records')
eval_data = eval_df.to_dict(orient='records')

# Save the dictionaries as .pkl files
with open('./[PATH AND FILENAME TO SAVE TRAIN DATASET].pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('./[PATH AND FILENAME TO SAVE EVAL DATASET].pkl', 'wb') as f:
    pickle.dump(eval_data, f)
