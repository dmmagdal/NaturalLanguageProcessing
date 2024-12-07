import torch
import re

# Step 1: Define stop words (expand this as needed)
STOP_WORDS = set([
    "a", "an", "the", "and", "but", "or", "on", "in", "with", "by", "to", "for", "from", 
    "about", "as", "at", "into", "through", "between", "during", "before", "after"
])

# Step 2: Preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    words = text.split()  # Split into words
    words = [word for word in words if word not in STOP_WORDS]  # Remove stopwords
    return words

# Step 3: Generate candidate keywords (unigrams and n-grams)
def generate_candidates(words, max_ngram=3):
    candidates = []
    for i in range(len(words)):
        for n in range(1, max_ngram + 1):
            if i + n <= len(words):
                candidates.append(' '.join(words[i:i + n]))
    return candidates

# Step 4: Calculate keyword score using YAKE's scoring formula (optimized with PyTorch)
def calculate_yake_score(word, candidates, words):
    # Efficient frequency calculation using PyTorch
    words_tensor = torch.tensor([hash(word) % 100000 for word in words])  # Hash words to indices
    # f_w = (words_tensor == word).sum().item()  # Frequency of the word in the candidates

    word_tensor = torch.tensor([hash(word) % 100000])   # Hash the target word to compare
    f_w = torch.sum(words_tensor == word_tensor).item()  # Frequency of the word in the candidates
    
    # Handle zero frequency and calculate probability
    p_w = f_w / len(words) if f_w > 0 else 1e-6  # Avoid zero probability
    
    # Efficient distance calculation to the nearest stopword
    # word_positions = torch.where(words_tensor == word)[0]
    # stopword_positions = torch.where(torch.isin(words_tensor, torch.tensor(list(STOP_WORDS))))[0]
    word_positions = torch.where(words_tensor == word_tensor)[0]
    stopword_positions = torch.where(torch.isin(words_tensor, torch.tensor([hash(stop) % 100000 for stop in STOP_WORDS])))[0]
    
    if len(stopword_positions) > 0 and len(word_positions) > 0:
        d_w = torch.min(torch.abs(word_positions[:, None] - stopword_positions)).item()  # Min distance to stopword
    else:
        d_w = 1e-6  # If no stopword is found, set a small constant
    
    # YAKE score formula (avoid zero division by adding small constants)
    score = f_w / ((d_w + 1e-6) * (p_w + 1e-6))  # Adding small constants to avoid zero division
    
    return score

# Step 5: Extract top N keywords
def yake(text, top_n=3, max_ngram=3):
    words = preprocess_text(text)  # Preprocess the text
    candidates = generate_candidates(words, max_ngram)  # Generate candidate keywords
    
    # Efficient calculation of scores using PyTorch
    scores = {candidate: calculate_yake_score(candidate, candidates, words) for candidate in candidates}
    
    # Sort candidates by score in descending order using PyTorch (similar to NumPy)
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [candidate for candidate, score in sorted_candidates[:top_n]]
    
    return top_keywords

# Example usage
sentence = "The quick brown fox jumps over the lazy dog near the river."
sentence = "Who ran in the 1938 Olympic games for America?"
sentence = "What color are sapphires?"
keywords = yake(sentence, top_n=3)
print("Top keywords:", keywords)
