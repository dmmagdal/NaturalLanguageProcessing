import torch
import re

# Step 1: Define stop words (you can expand this list)
STOP_WORDS = set([
    "a", "an", "the", "and", "but", "or", "on", "in", "with", "by", "to", "for", "from", 
    "about", "as", "at", "into", "through", "between", "during", "before", "after"
])

# Step 2: Preprocess the sentence (remove punctuation and lowercase)
def preprocess_sentence(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())  # Remove punctuation
    words = sentence.split()  # Split into words
    words = [word for word in words if word not in STOP_WORDS]  # Remove stop words
    return words

# Step 3: Calculate word frequency using PyTorch
def calculate_word_frequency(words):
    # Convert list of words to a tensor of indices
    word_tensor = torch.tensor([hash(word) % 100000 for word in words])  # Use a hash to map words to indices
    
    # Count occurrences of each word using PyTorch's unique function
    unique_words, counts = torch.unique(word_tensor, return_counts=True)
    
    # Convert to a dictionary: unique word hash -> count
    word_freq = {unique_words[i].item(): counts[i].item() for i in range(len(unique_words))}
    
    return word_freq, unique_words, counts

# Step 4: Extract top-N keywords
def extract_top_keywords(sentence, top_n=3):
    words = preprocess_sentence(sentence)
    word_freq, unique_words, counts = calculate_word_frequency(words)
    
    # Convert the unique word indices to actual words
    word_list = [words[i] for i in range(len(words)) if hash(words[i]) % 100000 in word_freq]
    
    # Sort based on counts (frequency) in descending order and get top-N keywords
    sorted_indices = torch.argsort(counts, descending=True)[:top_n]
    
    top_keywords = [word_list[i] for i in sorted_indices]
    
    return top_keywords

# Example usage
sentence = "The quick brown fox jumps over the lazy dog near the river."
sentence = "Who ran in the 1938 Olympic games for America?"
sentence = "What color are sapphires?"
keywords = extract_top_keywords(sentence, top_n=3)
print("Top keywords:", keywords)