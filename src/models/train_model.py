import torch
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
from model import HeteroGCLSTM
from torch_geometric_temporal.signal import StaticHeteroGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import pickle

# Define the file path of the saved dictionaries
file_path = r'C:\Users\Dell\Documents\GitHub\pinns_opf\data\processed\dictionaries.pkl'

# Load the dictionaries from the file
with open(file_path, 'rb') as file:
    data_dict = pickle.load(file)

# Unpack the dictionaries
edge_index_dict = data_dict['edge_index_dict']
feature_dicts = data_dict['feature_dicts']
target_dicts = data_dict['target_dicts']


# Define the file path of the saved metadata
file_path = r'C:\Users\Dell\Documents\GitHub\pinns_opf\data\processed\metadata.pkl'

# Load the metadata from the file
with open(file_path, 'rb') as file:
    metadata = pickle.load(file)
    
graph_snapshots = StaticHeteroGraphTemporalSignal(edge_index_dict, None, feature_dicts, target_dicts)

train_dataset, test_dataset = temporal_signal_split(graph_snapshots, train_ratio=0.8)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


in_channels_dict = {
    "cons": 2,
    "nodes": 1  
}
    
h_dict = {
    "cons": 0,
    "nodes": 0
}

c_dict = {
    "cons": 2,
    "nodes": 1
}
out_channels = 64    
# Create an instance of the HeteroGCLSTM model
model = HeteroGCLSTM(in_channels_dict, out_channels, metadata).to(device)

# Set loss function and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0

    for time, snapshot in enumerate(train_dataset):
        snapshot = snapshot.to(device)

        optimizer.zero_grad()


        # Forward pass
        h_dict, c_dict = model(snapshot.x_dict, snapshot.edge_index_dict)

        # Compute loss
        loss = criterion(h_dict['cons'], snapshot.y['cons'])

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Compute average loss for the epoch
    avg_loss = total_loss / len(train_dataset)

    print(f'Epoch: {epoch + 1}, Loss: {avg_loss}')

# Evaluation loop
model.eval()
total_test_loss = 0

for time, snapshot in enumerate(test_dataset):
    data = snapshot.to(device)

    with torch.no_grad():
        h_dict, c_dict = model(snapshot.x_dict, snapshot.edge_index_dict)
        loss = criterion(h_dict['cons'], data.y['cons'])
        total_test_loss += loss.item()

# Compute average test loss
avg_test_loss = total_test_loss / len(test_dataset)
print(f'Test Loss: {avg_test_loss}')