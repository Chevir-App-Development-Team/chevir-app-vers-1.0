import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, epochs, learning_rate, device):
    """Tanıma modeli (Modül A) için eğitim döngüsü."""
    print(f"Eğitim başlıyor... Cihaz: {device}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Aşırı öğrenmeyi engelle
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            
        accuracy = 100 * correct / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    print("Eğitim tamamlandı.")
    return model