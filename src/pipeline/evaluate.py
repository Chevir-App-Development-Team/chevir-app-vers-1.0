import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import get_logger

logger = get_logger(__name__)

def evaluate_model(model, test_loader, device='cpu', class_names=None):
    """
    Evaluates the model on the test set and prints performance metrics.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    logger.info("--- Evaluation Results ---")
    logger.info("\nClassification Report:\n" + classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(all_labels, all_preds)))
    
    return all_preds, all_labels

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from src.module_a_s2t import SignLanguageLSTM
    from src.config import SEQUENCE_LENGTH, INPUT_SIZE, NUM_CLASSES
    
    logger.info("Running a mock evaluation loop...")
    
    # Mock data
    X_test = torch.randn(30, SEQUENCE_LENGTH, INPUT_SIZE)
    y_test = torch.randint(0, NUM_CLASSES, (30,))
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    model = SignLanguageLSTM(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    
    class_names = ["HELLO", "THANK YOU", "I LOVE YOU"]
    evaluate_model(model, test_loader, device='cpu', class_names=class_names)
