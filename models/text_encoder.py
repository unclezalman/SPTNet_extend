import torch
import clip

class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model_name="ViT-B/16", device="cuda"):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad=False
        
    def forward(self, text_inputs):
        if isinstance(text_inputs, list) or isinstance(text_inputs, str):
            text_inputs = clip.tokenize(text_inputs, truncate=True).to(self.device)
        # text_inputs shape: [batch_size, num_prompts, seq_len]
        if len(text_inputs.shape) == 3:  # [batch, num_prompts, seq_len]
            batch_size, num_prompts, seq_len = text_inputs.shape
            text_inputs = text_inputs.view(-1, seq_len)  # [batch*num_prompts, seq_len]
            
        # Get text features
        with torch.no_grad():  # Ensure no gradients
            text_features = self.clip_model.encode_text(text_inputs)
            
        # Reshape if we had multiple prompts
        if len(text_inputs.shape) == 3:
            text_features = text_features.view(batch_size, num_prompts, -1)
            
        return text_features
    
