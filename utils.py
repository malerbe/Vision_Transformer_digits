import torch
import os

def save_checkpoint(state, is_best, checkpoint_dir='./checkpoints'):
    """Sauvegarde un checkpoint (toujours last, si best marque aussi best)."""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, 'model_last.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filename)