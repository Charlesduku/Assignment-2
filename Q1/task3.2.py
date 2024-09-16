from transformers import AutoTokenizer, AutoModel
from collections import Counter
import torch
import warnings
warnings.filterwarnings("ignore")

#  Read the text file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Tokenize the text using AutoTokenizer
def tokenize_text(text, model_name='distilbert-base-uncased'):
    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    return tokens

# Count the occurrences of tokens
def count_tokens(tokens):
    return Counter(tokens)

# Get the Top 30 most common tokens
def get_top_n_tokens(counter, n=30):
    return counter.most_common(n)

# Main function to process the text and generate the top 30 tokens
def process_text_file(file_path, model_name='distilbert-base-uncased'):
    text = read_file(file_path)
    tokens = tokenize_text(text, model_name=model_name)
    token_count = count_tokens(tokens)
    top_30_tokens = get_top_n_tokens(token_count, n=30)
    
    return top_30_tokens

# Example usage
file_path = 'combined_texts.txt' 
top_30_tokens = process_text_file(file_path)

# Display the top 30 tokens
print("Top 30 tokens:")
for token, count in top_30_tokens:
    print(f"Token: {token}, Count: {count}")
