import re
from collections import Counter
import math

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

# Step 4: Calculate keyword score using YAKE's scoring formula
def calculate_yake_score(word, candidates, words):
    f_w = candidates.count(word)  # Frequency of the word in the candidates

    # Handle the case where the word frequency is 0 (to avoid division by zero in p_w)
    if f_w == 0:
        p_w = 1e-6  # A small constant added to avoid division by zero in p_w
    else:
        p_w = f_w / len(words)  # Probability of the word in the text

    # Calculate the distance to the nearest stopword
    try:
        d_w = min([abs(i - words.index(word)) for i in range(len(words)) if words[i] == word])
    except ValueError:
        d_w = 1e-6  # Small constant to avoid division by zero in d_w

    # d_w = min([abs(i - words.index(word)) for i in range(len(words)) if words[i] == word])  # Distance to nearest stopword
    # p_w = f_w / len(words)  # Probability of the word in the text
    # score = f_w / (d_w * p_w)  # YAKE score formula

    # To avoid division by zero in the score formula, we add a small constant to d_w and p_w
    score = f_w / ((d_w + 1e-6) * (p_w + 1e-6))  # Adding small constants to avoid zero division
    
    return score

# Step 5: Extract top N keywords
def yake(text, top_n=3, max_ngram=3):
    words = preprocess_text(text)  # Preprocess the text
    candidates = generate_candidates(words, max_ngram)  # Generate candidate keywords
    
    # Calculate scores for each candidate
    scores = {candidate: calculate_yake_score(candidate, candidates, words) for candidate in candidates}
    
    # Sort candidates by score in descending order and get top N
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [candidate for candidate, score in sorted_candidates[:top_n]]
    
    return top_keywords

# Example usage
sentence = "The quick brown fox jumps over the lazy dog near the river."
sentence = "Who ran in the 1938 Olympic games for America?"
sentence = "What color are sapphires?"
keywords = yake(sentence, top_n=3)
print("Top keywords:", keywords)
