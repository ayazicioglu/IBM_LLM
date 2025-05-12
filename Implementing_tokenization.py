import nltk
import spacy
from transformers import BertTokenizer, XLNetTokenizer
from datetime import datetime

# NLTK Tokenization
start_time = datetime.now()
nltk_tokens = nltk.word_tokenize(text)
nltk_time = datetime.now() - start_time

# SpaCy Tokenization
nlp = spacy.load("en_core_web_sm")
start_time = datetime.now()
spacy_tokens = [token.text for token in nlp(text)]
spacy_time = datetime.now() - start_time

# BertTokenizer Tokenization
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
start_time = datetime.now()
bert_tokens = bert_tokenizer.tokenize(text)
bert_time = datetime.now() - start_time

# XLNetTokenizer Tokenization
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
start_time = datetime.now()
xlnet_tokens = xlnet_tokenizer.tokenize(text)
xlnet_time = datetime.now() - start_time
    
# Display tokens, time taken for each tokenizer, and token frequencies
print(f"NLTK Tokens: {nltk_tokens}\nTime Taken: {nltk_time} seconds\n")
show_frequencies(nltk_tokens, "NLTK")

print(f"SpaCy Tokens: {spacy_tokens}\nTime Taken: {spacy_time} seconds\n")
show_frequencies(spacy_tokens, "SpaCy")

print(f"Bert Tokens: {bert_tokens}\nTime Taken: {bert_time} seconds\n")
show_frequencies(bert_tokens, "Bert")

print(f"XLNet Tokens: {xlnet_tokens}\nTime Taken: {xlnet_time} seconds\n")
show_frequencies(xlnet_tokens, "XLNet")