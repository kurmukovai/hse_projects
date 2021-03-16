import torch

def train_step(images, labels, architecture, criterion, optimizer):
    device = ("cuda" if torch.cuda.is_available() else 'cpu')
    images, labels = images.to(device), labels.to(device)
    architecture.train() # enforce training regime
    
    pred_logits = architecture(images)
    loss = criterion(pred_logits, labels)
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.data.cpu().numpy()
