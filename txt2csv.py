import csv
import glob
import os
import tkinter as tk
from tkinter import ttk
import chardet
from tkinter import messagebox
from collections import defaultdict

def convert_to_csv():
    # Update the root directory path if needed
    root_directory = ""

    input_files = sorted(glob.glob(os.path.join(root_directory, "/input/Conversation*.txt")))
    output_folder = os.path.join(root_directory, "converted")
    delimiter = ','

    if not input_files:
        status_label.config(text="Error: No input files found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_skipped_rows = 0
    skipped_rows_by_dataset = defaultdict(int)
    total_rows_by_dataset = defaultdict(int)

    start_button.config(state=tk.DISABLED)

    input_table.delete(*input_table.get_children())
    output_table.delete(*output_table.get_children())

    for idx, input_file in enumerate(input_files, start=1):
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".csv")

        with open(input_file, 'rb') as file:
            raw_data = file.read()
            encoding_result = chardet.detect(raw_data)
            file_encoding = encoding_result['encoding']
            lines = raw_data.decode(file_encoding).splitlines()

        data = []
        skipped_rows = 0

        for line in lines:
            total_rows_by_dataset[input_file] += 1

            try:
                items = line.split('\t')
                data.append(items)
            except UnicodeEncodeError:
                print(f"Skipped row in file '{input_file}': {line}")
                skipped_rows += 1
                skipped_rows_by_dataset[input_file] += 1
                continue

        total_skipped_rows += skipped_rows

        status_label.config(text=f"Conversion in progress: {idx}/{len(input_files)} files converted")

        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerows(data)

        # Update the tables
        input_table.insert("", "end", values=(input_file,))
        output_table.insert("", "end", values=(output_file,))

    status_label.config(text=f"Conversion complete! Files stored in '{output_folder}' folder.")
    start_button.config(text="Close")
    start_button.config(command=window.quit)
    start_button.config(state=tk.NORMAL)

    for dataset, total_rows in total_rows_by_dataset.items():
        skipped_rows = skipped_rows_by_dataset[dataset]
        dataset_percentage = (skipped_rows / total_rows) * 100
        messagebox.showinfo("Conversion Summary", f"Percentage of skipped rows in {dataset}: {dataset_percentage:.2f}%")

# Create a Tkinter window
window = tk.Tk()
window.title("Text to CSV Converter")

# Create a label for status messages
status_label = tk.Label(window, text="Click below to convert!")
status_label.pack(pady=5)

# Create a button to start the conversion
start_button = ttk.Button(window, text="Start Conversion", command=convert_to_csv)
start_button.pack(pady=10)

# Create input table
input_table = ttk.Treeview(window, columns=("Input Files",), show="headings")
input_table.heading("Input Files", text="Input Files")
input_table.pack(side=tk.LEFT, padx=10, pady=10)

# Create output table
output_table = ttk.Treeview(window, columns=("Output Files",), show="headings")
output_table.heading("Output Files", text="Output Files")
output_table.pack(side=tk.LEFT, padx=10, pady=10)

# Run the Tkinter event loop
window.mainloop()
