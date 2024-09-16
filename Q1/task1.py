import pandas as pd

# File paths for CSV files
file_paths = [
    'CSV1.csv',
    'CSV2.csv',
    'CSV3.csv',
    'CSV4.csv',

]
# Initialize a list to store the extracted text data
texts = []

# Loop through each file, load the CSV, and extract the 'text' column
for file_path in file_paths:
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Extract 'text' column if it exists in the CSV
    if 'TEXT' in df.columns:
        texts.extend(df['TEXT'].dropna().tolist())

# Combine all the extracted texts into one string, each separated by a newline
combined_text = "\n".join(texts)


# Define the output path for the combined text file
txt_output_path = 'combined_texts.txt'

# Write the combined text to the output file
with open(txt_output_path, 'w') as f:
    f.write(combined_text)

# Output the path of the generated file
print(f"Combined text file saved at: {txt_output_path}")

