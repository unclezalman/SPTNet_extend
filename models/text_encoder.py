import torch
import clip
from PIL import Image

class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model_name="ViT-B/16", device="cuda"):
        super().__init__()
        self.device = device
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.clip_model = self.clip_model.eval()
        
        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, text_inputs):
        """
        Handle different input types:
        - List of strings: ["a cat", "a dog"]
        - String: "a photo of a cat"
        - Pre-tokenized tensor: [batch_size, seq_len] or [batch_size, num_prompts, seq_len]
        """
        # Handle raw text input
        if isinstance(text_inputs, (list, str)):
            text_inputs = clip.tokenize(text_inputs, truncate=True).to(self.device)
            
        # Handle multiple prompts per sample
        original_shape = text_inputs.shape
        if len(original_shape) == 3:
            batch_size, num_prompts, seq_len = original_shape
            text_inputs = text_inputs.view(-1, seq_len).to(self.device)
            
        # Get text features
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            print(f"Text features shape: {text_features.shape}")
            
        # Reshape if needed
        if len(original_shape) == 3:
            text_features = text_features.view(batch_size, num_prompts, -1)
            
        return text_features
