import os
import pickle

# Define the file path
file_path = './[PATH TO YOUR FILE].pkl'

# Extract the file name without the path
file_name = os.path.basename(file_path)

# Load the dataset from the .pkl file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Extract the column names from the dataset
column_names = list(data[0].keys())

# Print the number of items in each column
print("Items per column in '{}':".format(file_name))
for column_name in column_names:
    column_items = [record[column_name] for record in data]
    num_items = len(column_items)
    print("{}: {}".format(column_name, num_items))

# Print the total number of columns
num_columns = len(column_names)
print("Total number of columns in '{}': {}".format(file_name, num_columns))
