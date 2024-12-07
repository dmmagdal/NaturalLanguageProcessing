import torch
# import numpy as np
import string

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "it", "to", "of", "in", "for", "on", "with", "by", "as", "that", "at", "which", "be", "was", "were", "has", "have", "had", "having"
}

def preprocess_sentence(sentence):
    sentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    words = [word for word in sentence.split() if word not in STOP_WORDS]
    return words

def build_graph_pytorch(tokens, window_size=2):
    word_to_index = {word: idx for idx, word in enumerate(set(tokens))}
    n = len(word_to_index)
    graph = torch.zeros((n, n), dtype=torch.float32)

    for i, word in enumerate(tokens):
        word_idx = word_to_index[word]
        for j in range(i + 1, min(i + window_size + 1, len(tokens))):
            neighbor_idx = word_to_index[tokens[j]]
            if word_idx != neighbor_idx:
                graph[word_idx, neighbor_idx] += 1
                graph[neighbor_idx, word_idx] += 1

    return graph, word_to_index

def textrank_pytorch(graph, num_iterations=10, d=0.85):
    scores = torch.ones(graph.shape[0], dtype=torch.float32)
    
    for _ in range(num_iterations):
        new_scores = (1 - d) + d * torch.matmul(graph, scores) / graph.sum(dim=1)
        scores = new_scores
    
    return scores

# Example usage with PyTorch
def extract_keywords(sentence, top_n=5):
    tokens = preprocess_sentence(sentence)
    
    # Build the graph and get the word-to-index mapping
    graph, word_to_index = build_graph_pytorch(tokens)
    
    # Apply the TextRank algorithm
    scores = textrank_pytorch(graph)
    
    # Map the scores back to words
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    
    # Get the indices of the top-N scores
    top_indices = torch.argsort(scores, descending=True)[:top_n]
    
    # Get the corresponding words for the top-N indices
    top_keywords = [index_to_word[idx.item()] for idx in top_indices]
    
    return top_keywords

sentence = "The quick brown fox jumps over the lazy dog near the river."
sentence = "Who ran in the 1938 Olympic games for America?"
sentence = "What color are sapphires?"
keywords = extract_keywords(sentence, top_n=3)
print("Top keywords:", keywords)