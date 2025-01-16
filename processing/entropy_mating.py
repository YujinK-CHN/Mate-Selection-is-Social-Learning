import torch

# Define the entropy calculation function
def entropy(values):
    if len(values) == 0:
        return 0.0
    prob_dist = values / values.sum()
    return -torch.sum(prob_dist * torch.log(prob_dist + 1e-10))

# Split fitness into worse, similarity, and better parts
def split_fitness_parts(fitness1, fitness2, eta):
    worse_part = fitness1[fitness1 < fitness2 - eta]
    similarity_part = fitness1[torch.abs(fitness1 - fitness2) <= eta]
    better_part = fitness1[fitness1 > fitness2 + eta]
    return worse_part, similarity_part, better_part

# Calculate the symmetric evaluation score
def symmetric_evaluation(worse, similarity, better, N):
    W, S, B = len(worse), len(similarity), len(better)
    return torch.log(torch.tensor(N, dtype=torch.float32)) - (1/N) * (
        W * torch.log(torch.tensor(W + 1e-10)) + 
        S * torch.log(torch.tensor(S + 1e-10)) + 
        B * torch.log(torch.tensor(B + 1e-10))
    )

# Calculate the score for each agent pair using the defined method
def entropy_activated_score(fitness1, fitness2, eta):
    worse, similarity, better = split_fitness_parts(fitness1, fitness2, eta)
    N = len(fitness1)  # Number of fitness values
    return symmetric_evaluation(worse, similarity, better, N)

# Calculate the bidirectional score matrix
def bi_directional_score_matrix(fitness_vectors, eta):
    pop_size = fitness_vectors.shape[0]
    score_matrix = torch.zeros(pop_size, pop_size)

    for i in range(pop_size):
        for j in range(pop_size):
            if i != j:
                score_matrix[i, j] = entropy_activated_score(fitness_vectors[i], fitness_vectors[j], eta)
                score_matrix[j, i] = score_matrix[i, j]  # Symmetric score

    return score_matrix

# Convert scores into probability distributions using softmax
def probability_distribution(score_matrix):
    return torch.softmax(score_matrix, dim=1)

# Sample mates based on the probability distribution
def sample_mates(population, prob_matrix):
    pop_size = prob_matrix.shape[0]
    mates = []

    # Step 1: Set diagonal to zero to prevent self-mating
    prob_matrix.fill_diagonal_(0)

    # Step 2: Sample mates
    for i in range(pop_size):
        mate_index = torch.multinomial(prob_matrix[i], 1).item()
        mates.append(population[mate_index])
    
    return mates