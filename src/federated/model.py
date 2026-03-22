# src/federated/model.py

# torch is PyTorch - the most popular AI/ML library
# nn means Neural Network - it helps us define AI models
import torch
import torch.nn as nn

# SentenceTransformer is the same model your project already uses
# to convert text into numbers (embeddings)
from sentence_transformers import SentenceTransformer

# OrderedDict is like a regular dictionary but keeps the order
# We use it to store model weights in correct order
from collections import OrderedDict

# List and Tuple are just type hints
# They help us know what type of data a function expects
from typing import List, Tuple


# This is our main class for the Federated Model
# Think of a class as a blueprint for creating objects
class FederatedEmbeddingModel(nn.Module):
    
    # __init__ is called when we create the model
    # model_name is which sentence transformer to use
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        
        # This line is required when using PyTorch nn.Module
        # It initializes the parent class
        super(FederatedEmbeddingModel, self).__init__()
        
        # Load the same model your project already uses
        # This converts sentences into 384 numbers
        self.encoder = SentenceTransformer(model_name)
        
        # Store the model name for later use
        self.model_name = model_name
        
        # This is a small neural network layer on top
        # 384 → 384 means input 384 numbers, output 384 numbers
        # This layer will be trained in Federated Learning
        self.projection = nn.Linear(384, 384)
        
        # ReLU is an activation function
        # It adds non-linearity to the model
        # Simply: it helps the model learn complex patterns
        self.relu = nn.ReLU()

    # This function converts text to numbers (embeddings)
    # texts is a list of strings like ["hello", "world"]
    def encode(self, texts: List[str]) -> torch.Tensor:
        
        # First use the sentence transformer to get embeddings
        # show_progress_bar=False means don't show loading bar
        embeddings = self.encoder.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Pass through our projection layer
        # This is the part that gets trained in FL
        output = self.projection(embeddings)
        
        # Apply ReLU activation
        output = self.relu(output)
        
        return output

    # get_weights extracts all the trainable numbers from the model
    # These weights are what we SHARE in Federated Learning
    # NOT the actual PDF data — just the learned knowledge
    def get_weights(self) -> List[torch.Tensor]:
        
        # state_dict() returns all model parameters as a dictionary
        # .values() gets just the values (the actual numbers)
        return [val.cpu().numpy() for val in self.state_dict().values()]

    # set_weights updates the model with new weights
    # This is called when server sends back improved weights
    def set_weights(self, weights: List) -> None:
        
        # Get the current parameter names
        params_dict = zip(self.state_dict().keys(), weights)
        
        # Create a new state dictionary with updated weights
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        
        # Load the new weights into the model
        # strict=True means all weights must match exactly
        self.load_state_dict(state_dict, strict=True)


# This function creates the model on the correct device
# MacBook M4 has MPS (Metal Performance Shaders) - Apple's GPU
def get_model() -> FederatedEmbeddingModel:
    
    # Check if M4 GPU (MPS) is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MacBook M4 GPU (MPS)")
    
    # If not, check for NVIDIA GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    
    # Otherwise use CPU
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create the model and move it to the correct device
    model = FederatedEmbeddingModel()
    model = model.to(device)
    
    return model