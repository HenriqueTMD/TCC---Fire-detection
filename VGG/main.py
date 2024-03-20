import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rotate

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG19'])

        # Calculate the flattened size based on the dimensions of the feature maps
        self.flattened_size = self.calculate_flattened_size()

        self.fcs = nn.Sequential(
            nn.Linear(self.flattened_size, 4096),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flattening the output of convolutional layers
        x = self.fcs(x)
        return x

    def calculate_flattened_size(self):
        # Assuming input size of (3, 224, 224)
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, 224, 224)
            features = self.conv_layers(dummy_input)
            return features.view(1, -1).shape[1]

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root, cls)
            class_idx = self.class_to_idx[cls]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, class_idx))
        return images

    def __len__(self):
        return len(self.images) * 4  # Each image will be rotated 4 times

    def __getitem__(self, idx):
        img_idx = idx // 4  # Index of the original image
        rotation_angle = idx % 4  # 0, 1, 2, or 3 for each rotation (0°, 90°, 180°, 270°)

        img_path, class_idx = self.images[img_idx]
        image = Image.open(img_path).convert('RGB')

        # Rotate the image based on rotation_angle
        rotated_image = rotate(image, rotation_angle * 90)

        if self.transform:
            rotated_image = self.transform(rotated_image)
        return rotated_image, class_idx

def main():
    device = torch.device("cuda")
    print(f'Using device: {device}')

    # Define dataset directory
    data_dir = 'C:\\Users\\henri\\Desktop\\tcc\\Sets'

    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom datasets for train, validation, and test
    train_dataset = CustomDataset(root=os.path.join(data_dir, 'train'), transform=transform)
    valid_dataset = CustomDataset(root=os.path.join(data_dir, 'valid'), transform=transform)
    test_dataset = CustomDataset(root=os.path.join(data_dir, 'test'), transform=transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize VGG model and move it to GPU
    model = VGG_net(in_channels=3, num_classes=1000).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_losses, valid_losses, train_accuracies, valid_accuracies = train(model, criterion, optimizer, train_loader, valid_loader, epochs=10, device=device)

    # Plot loss and accuracy
    plot_graph(train_losses, valid_losses, 'Loss')
    plot_graph(train_accuracies, valid_accuracies, 'Accuracy', acc=True)

    # Evaluate the model on the test set
    test_accuracy = evaluate(model, criterion, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')

def train(model, criterion, optimizer, train_loader, valid_loader, epochs=10, device='cpu'):
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(epochs):
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

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validate the model
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss /= len(valid_loader)
        valid_accuracy = correct / total
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

    return train_losses, valid_losses, train_accuracies, valid_accuracies

def plot_graph(train_values, valid_values, title, acc=False):
    plt.figure()
    plt.plot(train_values, label='Training')
    plt.plot(valid_values, label='Validation')
    plt.title(f'{title} vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate(model, criterion, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    return test_accuracy


if __name__ == "__main__":
    main()