import torch
import torch.nn as nn
import torch.optim as optim
from model import TamilNet
from preprocess import get_data_loaders

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TamilNet(num_classes=156).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = get_data_loaders(
        train_dir=r'D:\projects ai\tamil-handwritten-recognition\data\train',
        test_csv=r'D:\projects ai\tamil-handwritten-recognition\data\test.csv',
        test_dir=r'D:\projects ai\tamil-handwritten-recognition\data\test',
        batch_size=32
    )

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            if images is None or labels is None:  # Skip empty batches
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/batch_count:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], No valid batches processed")

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if images is None or labels is None:  # Skip empty batches
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total > 0:
            print(f"Test Accuracy: {100 * correct / total:.2f}%")
        else:
            print(f"Test Accuracy: N/A (no valid test batches)")

    torch.save(model.state_dict(), r'D:\projects ai\tamil-handwritten-recognition\data\tamilnet_model.pt')
    print("Model saved to data/tamilnet_model.pt")

if __name__ == "__main__":
    train_model()