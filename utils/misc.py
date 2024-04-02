import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def train(model, device, X, Y, class_weights=None, lr=0.001, batch_size=256, num_epoch=16, end_factor=0.1, use_tqdm=True):    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=num_epoch*len(X)//batch_size)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    
    if use_tqdm:
        pbar = tqdm(range(0, num_epoch))
    else:
        pbar = range(0, num_epoch)
    for epoch in pbar:
        for batch_idx in range(0, len(X), batch_size):
            data = X[batch_idx:batch_idx+batch_size]
            target = Y[batch_idx:batch_idx+batch_size]

            data, target = torch.tensor(data).to(device), torch.tensor(target).to(device).long()

            data = data.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if use_tqdm:
                pbar.set_description(f"loss: {loss.item():.5f}")
    
    return model

def test(model, device, X):
    model.eval()
    with torch.no_grad():
        probs = []
        for i in range(0, len(X)):
            data = X[i]
            
            data = torch.tensor(data).to(device)
            data = data.unsqueeze(0).unsqueeze(1)
            
            output = model(data).softmax(dim=-1)
            probs.append(output.cpu().numpy())

    probs = np.concatenate(probs, axis=0)

    return probs