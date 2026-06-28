import torch
import logging
logger = logging.getLogger(__name__)
import torch.nn.functional as F
import copy
from typing import Dict, Any, Tuple

def train_epoch(model: torch.nn.Module, train_loader: 'torch_geometric.loader.DataLoader', optimizer: torch.optim.Optimizer, device: torch.device, physics_weight: float = 0.1) -> float:
    logger.debug('Starting train epoch')
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # Main loss (MSE)
        mse_loss = F.mse_loss(pred.view(-1, 1), data.y)

        # Add physics-based regularization
        physics_reg = model.physics_regularization(data, pred)
        loss = mse_loss + physics_weight * physics_reg

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_nodes

    return total_loss / len(train_loader.dataset)

def evaluate(model: torch.nn.Module, loader: 'torch_geometric.loader.DataLoader', device: torch.device) -> Tuple[float, float, float]:
    logger.debug('Evaluating model')
    model.eval()
    mse_sum = 0
    mae_sum = 0
    num_nodes = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)

            mse = F.mse_loss(pred.view(-1, 1), data.y, reduction='sum').item()
            mse_sum += mse

            mae = F.l1_loss(pred.view(-1, 1), data.y, reduction='sum').item()
            mae_sum += mae

            num_nodes += data.num_nodes

    mse_avg = mse_sum / num_nodes
    mae_avg = mae_sum / num_nodes
    rmse_avg = mse_avg ** 0.5

    return mse_avg, mae_avg, rmse_avg

def train_model(model: torch.nn.Module, train_loader: 'torch_geometric.loader.DataLoader', val_loader: 'torch_geometric.loader.DataLoader', device: torch.device, num_epochs: int = 100, lr: float = 0.001, patience: int = 15) -> Dict[str, Any]:
    logger.info(f'Starting training for {num_epochs} epochs')
    """Train the model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mse, _, _ = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_mse)

        scheduler.step(val_mse)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history
