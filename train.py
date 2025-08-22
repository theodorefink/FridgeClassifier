from loader import train_loader, validate_loader
from model import model, criterion, optimizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")
model = model.to(device)

num_epochs = 5

for epochs in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f'Epoch [{epochs+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')

    # Validation phase
    model.eval()
    correct = 0
    total = 0  
    with torch.no_grad():
        for images, labels in validate_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%')

    # Save the model 
torch.save(model.state_dict(), "fridge_resnet18.pth")
print("Model saved as fridge_resnet18.pth")
