import torch

def train_step(images, labels, architecture, criterion, optimizer):
    device = ("cuda" if torch.cuda.is_available() else 'cpu')
    images, labels = images.to(device).float(), labels.to(device).float()
    architecture.train()
    optimizer.zero_grad()
    pred_logits = architecture(images)
    loss = criterion(pred_logits, labels)
    loss.backward()
    optimizer.step()
    
    return loss.data.cpu().numpy()