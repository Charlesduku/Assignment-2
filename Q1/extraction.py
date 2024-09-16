# Import necessary libraries
import spacy
import scispacy
from scispacy.linking import EntityLinker
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter
from huggingface_hub import login

# Load SciSpaCy models
nlp_scispacy = spacy.load("en_core_sci_sm")
nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")

# Function to extract disease and drug entities from a model
def extract_entities_sci(text, model):
    doc = model(text)
    entities = {'drugs': [], 'diseases': []}
    
    for ent in doc.ents:
        if ent.label_ == 'DISEASE':
            entities['diseases'].append(ent.text)
        elif ent.label_ == 'CHEMICAL':
            entities['drugs'].append(ent.text)
    
    return entities

# Load BioBERT model for NER
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
nlp_biobert = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to extract disease and drug entities using BioBERT
def extract_entities_biobert(text):
    ner_results = nlp_biobert(text)
    entities = {'drugs': [], 'diseases': []}
    
    for result in ner_results:
        if 'drug' in result['entity'].lower():
            entities['drugs'].append(result['word'])
        elif 'disease' in result['entity'].lower():
            entities['diseases'].append(result['word'])
    
    return entities

# Function to read text from file
def extract_entities_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Extract entities using SciSpaCy and BioBERT
    entities_scispacy = extract_entities_sci(text, nlp_scispacy)
    entities_bc5cdr = extract_entities_sci(text, nlp_bc5cdr)
    entities_biobert = extract_entities_biobert(text)
    
    return entities_scispacy, entities_bc5cdr, entities_biobert

# Function to compare entities between models
def compare_entities(entities_scispacy, entities_bc5cdr, entities_biobert):
    # Calculate total entities detected by each model
    total_scispacy = sum(len(entities_scispacy[key]) for key in entities_scispacy)
    total_bc5cdr = sum(len(entities_bc5cdr[key]) for key in entities_bc5cdr)
    total_biobert = sum(len(entities_biobert[key]) for key in entities_biobert)
    
    print(f"Total entities detected by en_core_sci_sm: {total_scispacy}")
    print(f"Total entities detected by en_ner_bc5cdr_md: {total_bc5cdr}")
    print(f"Total entities detected by BioBERT: {total_biobert}")
    
    # Find common and different entities between models
    common_diseases = set(entities_bc5cdr['diseases']).intersection(entities_biobert['diseases'])
    common_drugs = set(entities_bc5cdr['drugs']).intersection(entities_biobert['drugs'])

    different_diseases = set(entities_bc5cdr['diseases']).difference(entities_biobert['diseases'])
    different_drugs = set(entities_bc5cdr['drugs']).difference(entities_biobert['drugs'])

    print(f"\nCommon diseases detected by both BC5CDR and BioBERT: {common_diseases}")
    print(f"Different diseases detected by BC5CDR and BioBERT: {different_diseases}")
    
    print(f"\nCommon drugs detected by both BC5CDR and BioBERT: {common_drugs}")
    print(f"Different drugs detected by BC5CDR and BioBERT: {different_drugs}")

    # Most common entities in both models
    most_common_bc5cdr = Counter(entities_bc5cdr['diseases'] + entities_bc5cdr['drugs']).most_common(10)
    most_common_biobert = Counter(entities_biobert['diseases'] + entities_biobert['drugs']).most_common(10)

    print(f"\nMost common entities detected by BC5CDR: {most_common_bc5cdr}")
    print(f"Most common entities detected by BioBERT: {most_common_biobert}")

# Main function to run the extraction and comparison
if __name__ == '__main__':
    # Add your Hugging Face token here if required (for BioBERT)
    HF_TOKEN = "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"  # Replace with your Hugging Face API token if necessary
    
    # Path to your text file
    file_path = 'combined_texts.txt'  # Replace with the path to your .txt file
    
    # Extract entities from both models
    entities_scispacy, entities_bc5cdr, entities_biobert = extract_entities_from_text(file_path)
    
    # Compare the extracted entities
    compare_entities(entities_scispacy, entities_bc5cdr, entities_biobert)
