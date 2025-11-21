import torch

def evaluate(model, data_loader, metric, device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)

            metric.update(y_pred, y_batch)

    return metric


def train(model, optimizer, criterion, metric, train_loader, valid_loader, n_epochs, device):
    
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    
    for epoch in range(n_epochs):
        metric.reset()
        total_loss = 0
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            total_loss += loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            metric.update(y_pred, y_batch)

        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss.item())
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(evaluate(model, valid_loader, metric, device).compute().item())
        
        
        print(f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {history['train_losses'][-1]:.4f}, "
            f"train metric: {history['train_metrics'][-1]:.4f}, "
            f"valid metric: {history['valid_metrics'][-1]:.4f}")