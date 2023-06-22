import pandas as pd
import pickle

# Load the dataset from a CSV file
df = pd.read_csv('./datasets/chocolate_makers.csv')

# Extract the labels from a unique column
labels = df['COMPANY NAME']

# Clean and normalize the data (if required)
# No specific cleaning or normalization steps mentioned in the dataset structure

# Rename the columns
df = df.rename(columns={
    'COMPANY NAME': 'labels',
    'CITY': 'City',
    'STATE/PROVINCE': 'State',
    'OWNER/MAKER': 'Owner',
    'COUNTRY': 'Country'
})

# Split the dataset into training and evaluation sets (70% for training, 30% for evaluation)
train_df = df.sample(frac=0.7, random_state=1)
eval_df = df.drop(train_df.index)

# Convert the DataFrames to dictionaries
train_data = train_df.to_dict(orient='records')
eval_data = eval_df.to_dict(orient='records')

# Save the dictionaries as .pkl files
with open('./pickle/train/chocolate_makers-train.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('./pickle/eval/chocolate_makers-eval.pkl', 'wb') as f:
    pickle.dump(eval_data, f)
