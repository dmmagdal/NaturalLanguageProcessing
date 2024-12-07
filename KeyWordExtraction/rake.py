import re
from collections import defaultdict

# Step 1: List of stop words (can be extended or replaced with a comprehensive list)
STOP_WORDS = set([
    "a", "an", "the", "and", "but", "or", "on", "in", "with", "by", "to", "for", "from", 
    "about", "as", "at", "into", "through", "between", "during", "before", "after"
])

# Step 2: Preprocess the text (remove punctuation, lowercase, and split into sentences)
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split the text into sentences based on punctuation marks like period or exclamation mark
    sentences = re.split(r'[.?!]', text)
    # Clean and remove empty sentences after splitting
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

# Step 3: Build word frequency and co-occurrence graphs for sentence-level
def build_graph(sentences):
    word_freq = defaultdict(int)  # Word frequency
    word_cooc = defaultdict(lambda: defaultdict(int))  # Co-occurrence count
    
    for sentence in sentences:
        words = [word for word in sentence.split() if word not in STOP_WORDS]
        for word in words:
            word_freq[word] += 1
            for other_word in words:
                if word != other_word:
                    word_cooc[word][other_word] += 1
    
    return word_freq, word_cooc

# Step 4: Calculate word scores based on frequency and co-occurrence
def calculate_word_scores(word_freq, word_cooc):
    word_scores = {}
    for word in word_freq:
        cooc_sum = sum(word_cooc[word].values())
        word_scores[word] = word_freq[word] / (cooc_sum + 1)  # Avoid division by zero
    return word_scores

# Step 5: Score words in each sentence based on the sum of word scores in each sentence
def score_words_in_sentences(sentences, word_scores):
    sentence_keywords = {}
    
    for sentence in sentences:
        words = [word for word in sentence.split() if word not in STOP_WORDS]
        word_scores_in_sentence = [(word, word_scores.get(word, 0)) for word in words]
        # Sort words by their score
        sorted_word_scores = sorted(word_scores_in_sentence, key=lambda x: x[1], reverse=True)
        sentence_keywords[sentence] = sorted_word_scores
    
    return sentence_keywords

# Step 6: Rank the words within each sentence and extract the top-N keywords from each sentence
def extract_top_keywords(text, top_n=3):
    # Preprocess the text and split into sentences
    sentences = preprocess_text(text)
    
    # Build word frequency and co-occurrence graph for sentences
    word_freq, word_cooc = build_graph(sentences)
    
    # Calculate word scores
    word_scores = calculate_word_scores(word_freq, word_cooc)
    
    # Score the words within each sentence
    sentence_keywords = score_words_in_sentences(sentences, word_scores)
    
    # Extract the top-N keywords from each sentence
    top_keywords = {}
    for sentence, word_scores in sentence_keywords.items():
        top_keywords[sentence] = [word for word, score in word_scores[:top_n]]
    
    return top_keywords

# Example usage
# text = """
# The quick brown fox jumps over the lazy dog. 
# A lazy dog sits near the river. 
# The fox is quick and smart.
# """
# keywords_sentences = extract_keywords_sentences(text, top_n=3)
# print("Top keywords in sentences:", keywords_sentences)


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
