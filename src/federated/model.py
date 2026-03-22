import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from typing import List, Tuple
class FederatedEmbeddingModel(nn.Module):
    def __init__(self,model_name: str = "all-MiniLM-L6-v2"):
        super(FederatedEmbeddingModel, self).__init__()
        self.encoder = SentenceTransformer(model_name)
        self.model_name = model_name
        self.projection = nn.Linear(384, 384)
        self.relu = nn.RELU()
    def encode(self,texts: List[str]) -> torch.Tensor:
        embeddings =self.encoder.encode(
            text,
            convert_to_tensor=True
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