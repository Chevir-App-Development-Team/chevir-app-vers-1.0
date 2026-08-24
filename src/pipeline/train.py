import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, device='cpu'):
    """
    Standard PyTorch training loop with Adam, CrossEntropyLoss, and early stopping.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total if total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered. Halting training.")
            break
            
    return model

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from src.module_a_s2t.model import SignLanguageLSTM
    
    print("Running a mock training loop...")
    
    num_samples = 100
    seq_length = 30
    input_size = 1662
    num_classes = 3
    
    X_train = torch.randn(num_samples, seq_length, input_size)
    y_train = torch.randint(0, num_classes, (num_samples,))
    
    X_val = torch.randn(20, seq_length, input_size)
    y_val = torch.randint(0, num_classes, (20,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = SignLanguageLSTM(input_size=input_size, num_classes=num_classes)
    
    train_model(model, train_loader, val_loader, epochs=10, device='cpu')