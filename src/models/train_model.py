import os
import sys
import pickle
import logging
import time
import torch
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split

from model import RecurrentGCN


start_time = time.time()

#now we will Create and configure logger 
logging.basicConfig(filename="std.log", 
                format='%(asctime)s %(message)s', 
                filemode='w') 

#Let us Create an object 
logger=logging.getLogger()
# Set log level to DEBUG
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# GPU support
DEVICE = torch.device('cpu') # cuda
shuffle=True
batch_size = 32


# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the data from the pickle file
with open('../../data/processed/data.pickle', 'rb') as file:
    loaded_data = pickle.load(file)

# Access the loaded data
edge_index = loaded_data['edge_index']
edge_weight = loaded_data['edge_weight']
features_seq = loaded_data['features_seq']
targets_seq = loaded_data['targets_seq']


dataset = StaticGraphTemporalSignal(
    edge_index=edge_index, edge_weight=edge_weight,
    features=features_seq, targets=targets_seq
)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

model = RecurrentGCN(node_features=9)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model._train(model, train_dataset=train_dataset, epochs=10, lr=0.01, h=None, c=None)