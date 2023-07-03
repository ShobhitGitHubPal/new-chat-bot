# import torch
# from torch import nn

# input_dim = 1  # Number of rows in the embedding matrix
# embedding_dim = 1  # Dimensionality of the embedding vectors
# embedding = nn.Embedding(input_dim, embedding_dim)

# # Example input indices
# input_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# embedded_output = embedding(input_indices)
# print('kgjighhu',embedded_output)

# import torch

# def embedding(input_indices, embedding_dim):
#     # Define your embedding weight matrix
#     vocab_size = 1000  # Specify the size of your vocabulary
#     weight = torch.randn(vocab_size, embedding_dim)

#     # Retrieve the embeddings for the input indices
#     embeddings = weight[input_indices]

#     return embeddings


import torch

# Example embedding weight tensor
embedding_weights = torch.randn(100, 200)  # shape: (num_embeddings, embedding_dim)

# Example input indices
input_indices = torch.tensor([50, 30, 80])  # valid indices within the range of num_embeddings

# Ensure the input indices are within the valid range
if torch.max(input_indices) >= embedding_weights.size(0) or torch.min(input_indices) < 0:
    raise IndexError("Input indices are out of range")

# Perform embedding
embedded_values = torch.embedding(embedding_weights, input_indices)

print(embedded_values)