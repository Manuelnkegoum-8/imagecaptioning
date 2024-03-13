import torch
def train_epoch(model,dataloader,optimizer,criterion,device):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.
    model.train()
    for data in dataloader:
        images, texts = data
        bs = images.size(0)
        images = images.to(device)
        texts = texts.to(device)
        preds = model(images,texts[:, :-1])
        loss = criterion(preds.view(-1, preds.size(2)),texts[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss+=loss.item()*images.size(0)
        n+=images.size(0)
    del images,texts,data
    return avg_loss/n

@torch.no_grad()
def validate(model,dataloader,criterion,device):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.
    model.eval()
    for data in dataloader:
        images, texts = data
        bs = images.size(0)
        images = images.to(device)
        texts = texts.to(device)
        preds = model(images,texts[:, :-1])
        loss = criterion(preds.view(-1, preds.size(2)),texts[:, 1:].reshape(-1))
        avg_loss+=loss.item()*images.size(0)
        n+=images.size(0)
    del images,texts,data
    return avg_loss/n

    

