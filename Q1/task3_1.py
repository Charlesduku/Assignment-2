import csv
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
# Step 1: Read the text file
# file_path = "task1\combined_texts.txt"
# def read_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#     return text
# print(read_file)

nltk.download('stopwords')


# f = open("combined_texts.txt", "r")
# text = f.read()
# print(text)


def read_file(file_path):
    f= open(file_path, 'r')
    text = f.read()
    return text

# Clean the text and split into words

def clean_and_tokenize(text):
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize (split by whitespace)
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    return filtered_words



def count_words(words):
    return Counter(words)

# Get the Top 30 most common words
def get_top_n_words(counter, n=30):
    return counter.most_common(n)

#  Write the Top 30 words and their counts to a CSV file
def write_to_csv(word_counts, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Count'])  # Write header
        writer.writerows(word_counts)       # Write word counts


# Main function to process the text and generate the CSV
def process_text_file(file_path, output_csv):
    text = read_file(file_path)
    words = clean_and_tokenize(text)
    word_count = count_words(words)
    top_30_words = get_top_n_words(word_count, n=30)
    write_to_csv(top_30_words, output_csv)
    print(f"Top 30 words have been written to {output_csv}")


file_path = 'combined_texts.txt'  # file path
output_csv = 'top_30_words.csv'   # Name of the output CSV file

process_text_file(file_path, output_csv)

# print(read_file("combined_texts.txt"))

# print(clean_and_tokenize(texts))
