import string
import math

# Step 1: List of stop words for basic filtering
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "it", "to", "of", "in", "for", "on", "with", "by", "as", "that", "at", "which", "be", "was", "were", "has", "have", "had", "having"
}

# Step 2: Preprocessing the sentence
def preprocess_sentence(sentence):
    # Remove punctuation and convert to lowercase
    sentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    # Split sentence into words and remove stop words
    words = [word for word in sentence.split() if word not in STOP_WORDS]
    return words

# Step 3: Build a similarity graph (adjacency matrix for co-occurrence)
def build_graph(tokens, window_size=2):
    # Initialize an empty dictionary for the graph (using a dictionary of dictionaries)
    graph = {}
    
    # Loop over tokens and create edges based on co-occurrence within a sliding window
    for i, word in enumerate(tokens):
        if word not in graph:
            graph[word] = {}
        
        # Consider the neighboring words within the window size
        for j in range(i + 1, min(i + window_size + 1, len(tokens))):
            neighbor = tokens[j]
            if neighbor != word:
                # Add the neighbor to the current word's adjacency list
                if neighbor not in graph[word]:
                    graph[word][neighbor] = 0
                graph[word][neighbor] += 1
                
                # Ensure the reverse edge as well (since it's an undirected graph)
                if neighbor not in graph:
                    graph[neighbor] = {}
                if word not in graph[neighbor]:
                    graph[neighbor][word] = 0
                graph[neighbor][word] += 1
    
    return graph

# Step 4: TextRank Algorithm (Iterative ranking of words)
def textrank(graph, num_iterations=10, d=0.85):
    # Initialize the importance scores (PageRank-like scores)
    scores = {word: 1.0 for word in graph}
    
    for _ in range(num_iterations):
        new_scores = {}
        for word in graph:
            # Compute the sum of the importance scores of the neighboring words
            score = 0.0
            for neighbor, weight in graph[word].items():
                score += (scores[neighbor] * weight)
            new_scores[word] = (1 - d) + d * score
        
        # Update the scores
        scores = new_scores
    
    return scores

# Step 5: Extract the top-N keywords
def extract_keywords(sentence, top_n=5):
    # Preprocess the sentence
    tokens = preprocess_sentence(sentence)
    
    # Build the graph
    graph = build_graph(tokens)
    
    # Apply the TextRank algorithm
    scores = textrank(graph)
    
    # Sort the words by their importance scores in descending order
    sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top N keywords
    return [keyword for keyword, _ in sorted_keywords[:top_n]]

# Example usage
sentence = "The quick brown fox jumps over the lazy dog near the river."
sentence = "Who ran in the 1938 Olympic games for America?"
sentence = "What color are sapphires?"
keywords = extract_keywords(sentence, top_n=3)
print("Top keywords:", keywords)
