import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# Transform with data augmentation to help the model learn better
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # randomly flip images
    transforms.RandomRotation(10),     # randomly rotate images a bit
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # change lighting
    transforms.ToTensor(),
])

# Validation transform shouldn't have random changes
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset twice with different transforms
full_train_dataset = datasets.ImageFolder("data", transform=train_transform)
full_val_dataset = datasets.ImageFolder("data", transform=val_transform)

num_samples = len(full_train_dataset)
print(f"Total images found: {num_samples}")

# We need at least some data to train
if num_samples == 0:
    print("Error: No images found in the 'data' folder! Please run collect_data.py first.")
    exit()

# Figure out sizes for train and val split (80% train, 20% val)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

print(f"Training on {train_size} images, validating on {val_size} images.")

# Create random indices for splitting
indices = torch.randperm(num_samples).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create subsets
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# We can't shuffle validation data, it doesn't matter
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model setup
model = models.mobilenet_v2(weights=None)
# Get the number of classes from the dataset
num_classes = len(full_train_dataset.classes)
print("Classes:", full_train_dataset.classes)

model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_accuracy = 0.0

# Training loop
num_epochs = 10
print("Starting training...")

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train() # Set model to training mode
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # --- Validation Phase ---
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    
    # We don't need to calculate gradients for validation
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    avg_loss = total_loss / max(len(train_loader), 1)
    
    # Prevent divide by zero if val dataset is too small
    if total > 0:
        val_accuracy = 100 * correct / total
    else:
        val_accuracy = 0.0

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    # Save the model only if it gets better
    if val_accuracy >= best_val_accuracy and val_accuracy > 0:
        print(f"Validation accuracy improved! Saving model...")
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "model.pth")

print(f"Training complete! Best validation accuracy: {best_val_accuracy:.2f}%")