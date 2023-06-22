import os
import pickle

# Define the file path
file_path = './pickle/train/chocolate_makers-train.pkl'

# Extract the file name without the path
file_name = os.path.basename(file_path)

# Load the dataset from the .pkl file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Extract the column names from the dataset
column_names = list(data[0].keys())

# Print the column names
print("Column names in '{}':".format(file_name))
for column_name in column_names:
    print(column_name)

# Print the number of columns
num_columns = len(column_names)
print("Number of columns in '{}': {}".format(file_name, num_columns))
