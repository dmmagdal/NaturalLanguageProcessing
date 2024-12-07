import re
from collections import Counter

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

# Step 3: Calculate word frequency
def calculate_word_frequency(words):
    return Counter(words)

# Step 4: Extract top-N keywords
def extract_top_keywords(sentence, top_n=3):
    words = preprocess_sentence(sentence)
    word_freq = calculate_word_frequency(words)
    
    # Sort the words based on frequency in descending order
    sorted_keywords = word_freq.most_common(top_n)
    
    # Return the top-N keywords
    return [keyword for keyword, _ in sorted_keywords]

# Example usage
# text = """
# The quick brown fox jumps over the lazy dog. 
# A lazy dog sits near the river. 
# The fox is quick and smart.
# """

sentence = "The quick brown fox jumps over the lazy dog near the river."
sentence = "Who ran in the 1938 Olympic games for America?"
sentence = "What color are sapphires?"
keywords = extract_top_keywords(sentence, top_n=3)
print("Top keywords:", keywords)