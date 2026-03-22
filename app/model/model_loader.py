import torch
from app.config import settings
from app.model.model import AttentionNetwork

def load_model():
    model = AttentionNetwork(in_channels=1, num_classes=1)

    checkpoint = torch.load(
        settings.MODEL_PATH,
        map_location=torch.device('cpu')
    )

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(torch.device('cpu')) 
    model.eval()
    return model