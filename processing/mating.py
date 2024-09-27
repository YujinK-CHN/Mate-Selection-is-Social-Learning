import torch
import numpy as np

# Example fitness vectors with PyTorch
pop_size = 5
fitness_dim = 10
fitness_vectors = torch.rand(pop_size, fitness_dim)

# Cosine similarity function with PyTorch
def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))

# Superiority function (sum of fitness values ratio) using PyTorch
def superiority(fitness1, fitness2):
    sum_f1 = torch.sum(fitness1)
    sum_f2 = torch.sum(fitness2)
    return sum_f2 / sum_f1

# Combined score function (PyTorch) - alpha * similarity + beta * superiority
def combined_score(fitness1, fitness2, alpha=0.5, beta=0.5):
    sim = cosine_similarity(fitness1, fitness2)
    sup = superiority(fitness1, fitness2)
    return alpha * sim + beta * sup

# Pairwise score matrix calculation (PyTorch)
def pairwise_scores(fitness_vectors, alpha=0.5, beta=0.5):
    pop_size = fitness_vectors.shape[0]
    score_matrix = torch.zeros(pop_size, pop_size)

    for i in range(pop_size):
        for j in range(pop_size):
            if i != j:
                score_matrix[i, j] = combined_score(fitness_vectors[i], fitness_vectors[j], alpha, beta)

    return score_matrix

# Function to convert scores into probability distributions (PyTorch)
def probability_distribution(score_matrix):
    return torch.softmax(score_matrix, dim=1)



# Calculate pairwise scores and probabilities (PyTorch)
score_matrix = pairwise_scores(fitness_vectors)
prob_matrix = probability_distribution(score_matrix)

#print("PyTorch Scores Matrix:\n", score_matrix)
#print("PyTorch Probability Matrix:\n", prob_matrix)

# 'prob_matrix' contains the pairwise probabilities for mating
def sample_mates(pop, prob_matrix):
    pop_size = prob_matrix.shape[0]
    mates = []
    mate_indices = []

    # Step 1: Set diagonal to zero to prevent self-mating
    prob_matrix.fill_diagonal_(0)

    # Step 2: Compute bidirectional probabilities by multiplying P(a1 -> a2) * P(a2 -> a1)
    bidirectional_prob_matrix = prob_matrix * prob_matrix.T

    # Step 3: Normalize rows to ensure valid probability distributions
    row_sums = bidirectional_prob_matrix.sum(dim=1, keepdim=True)
    bidirectional_prob_matrix = bidirectional_prob_matrix / row_sums
    print("Prob matrix: \n", bidirectional_prob_matrix)
    # For each agent, sample a mate based on their probability distribution
    for i in range(pop_size):
        
        # Sample a mate index based on this distribution
        mate = torch.multinomial(bidirectional_prob_matrix[i], 1).item()
        
        mates.append(pop[mate])
        mate_indices.append(mate)
    
    return mates, mate_indices