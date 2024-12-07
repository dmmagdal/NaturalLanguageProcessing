import numpy as np
import string

# Step 1: List of stop words for basic filtering
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "it", "to", "of", "in", "for", "on", "with", "by", "as", "that", "at", "which", "be", "was", "were", "has", "have", "had", "having"
}

# Step 2: Preprocessing the sentence
def preprocess_sentence(sentence):
    sentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    words = [word for word in sentence.split() if word not in STOP_WORDS]
    return words

# Step 3: Build a graph (Adjacency matrix for co-occurrence)
def build_graph(tokens, window_size=2):
    # Create a mapping from word to index
    word_to_index = {word: idx for idx, word in enumerate(set(tokens))}
    n = len(word_to_index)  # Number of unique words
    graph = np.zeros((n, n), dtype=np.float32)  # Initialize the graph with zeros

    # Create co-occurrence matrix
    for i, word in enumerate(tokens):
        word_idx = word_to_index[word]
        for j in range(i + 1, min(i + window_size + 1, len(tokens))):
            neighbor_idx = word_to_index[tokens[j]]
            if word_idx != neighbor_idx:
                graph[word_idx, neighbor_idx] += 1
                graph[neighbor_idx, word_idx] += 1  # Ensure undirected edges

    return graph, word_to_index

# Step 4: TextRank Algorithm with NumPy (Efficient matrix operations)
def textrank(graph, num_iterations=10, d=0.85):
    # Initialize scores with 1.0 for all words
    scores = np.ones(graph.shape[0], dtype=np.float32)
    
    for _ in range(num_iterations):
        # Calculate the new scores as the weighted sum of neighbor scores
        new_scores = (1 - d) + d * np.dot(graph, scores) / np.sum(graph, axis=1)  # Normalize by the sum of edges
        
        # Update scores
        scores = new_scores
    
    return scores

# Step 5: Extract the top-N keywords
def extract_keywords(sentence, top_n=5):
    tokens = preprocess_sentence(sentence)
    
    # Build the graph and get the word-to-index mapping
    graph, word_to_index = build_graph(tokens)
    
    # Apply the TextRank algorithm
    scores = textrank(graph)
    
    # Map the scores back to words
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    
    # Get the indices of the top-N scores
    top_indices = np.argsort(scores)[::-1][:top_n]
    
    # Get the corresponding words for the top-N indices
    top_keywords = [index_to_word[idx] for idx in top_indices]
    
    return top_keywords

# Example usage
sentence = "The quick brown fox jumps over the lazy dog near the river."
sentence = "Who ran in the 1938 Olympic games for America?"
sentence = "What color are sapphires?"
keywords = extract_keywords(sentence, top_n=3)
print("Top keywords:", keywords)
