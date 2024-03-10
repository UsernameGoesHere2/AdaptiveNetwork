import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a simple dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the dataset into train and validation sets
train_X, val_X = X[:800], X[800:]
train_y, val_y = y[:800], y[800:]
class DynamicNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(DynamicNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = nn.ModuleList()

        # Initialize with a single hidden layer
        self.hidden_layers.append(nn.Linear(input_size, 10))

        self.output_layer = nn.Linear(10, output_size)

    def forward(self, x):
        out = x
        for layer in self.hidden_layers:
            out = torch.relu(layer(out))
        out = self.output_layer(out)
        return out

    def add_hidden_layer(self, input_size, output_size):
        new_layer = nn.Linear(input_size, output_size)
        self.hidden_layers.append(new_layer)

    def prune_hidden_layer(self, layer_index):
        self.hidden_layers.pop(layer_index)

# Initialize the network
net = DynamicNet(input_size=2, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
num_epochs = 200
best_val_loss = float('inf')
patience = 10
epochs_without_improvement = 0

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0

    # Training
    net.train()
    for X_batch, y_batch in zip(train_X.split(100), train_y.split(100)):
        optimizer.zero_grad()
        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    net.eval()
    with torch.no_grad():
        outputs = net(val_X)
        val_loss = criterion(outputs, val_y).item()

    # Print statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_X):.4f}, Val Loss: {val_loss:.4f}')

    # Add a new hidden layer if validation loss plateaus
    if val_loss >= best_val_loss:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            net.add_hidden_layer(net.hidden_layers[-1].out_features, 20)
            print(f'Added a new hidden layer with 20 neurons')
            best_val_loss = val_loss
            epochs_without_improvement = 0
    else:
        best_val_loss = val_loss

    # Prune a hidden layer if the network becomes too large
    if len(net.hidden_layers) > 5:
        net.prune_hidden_layer(0)
        print(f'Pruned the first hidden layer')
