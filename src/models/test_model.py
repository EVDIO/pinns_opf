import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import pickle
import matplotlib.pyplot as plt
from model import RecurrentGCN
import time

def load_data():
    # Load the data from the pickle file
    with open(r'C:\Users\edi\GitHub\pinns_opf\data\processed\data_node10.pickle', 'rb') as file:
        loaded_data = pickle.load(file)
        
    edge_index = loaded_data['edge_index']
    edge_weight = loaded_data['edge_weight']
    features_seq = loaded_data['features_seq']
    targets_seq = loaded_data['targets_seq']

    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index, edge_weight=edge_weight,
        features=features_seq, targets=targets_seq
    )

    _, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    
    return test_dataset


def evaluate_model(model, test_dataset):
    predictions, targets = model.evaluate(model, test_dataset)
    
    # Assuming the targets and predictions are 1-D, you can modify this if they aren't.
    plt.plot(targets, 'r', label='True Values')
    plt.plot(predictions, 'b', label='Predictions')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Comparison between True Values and Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()
    return predictions, targets





if __name__ == "__main__":
    # Load the test dataset
    test_dataset = load_data()

    # Load the saved model
    model_path = r"C:\Users\edi\GitHub\pinns_opf\src\models\model_final_pinns_14.pt"  # replace with your model path or provide a way to input it
    model = RecurrentGCN(node_features=12,k=14)
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    model.eval()
    predictions = []
    targets = []

    
    with torch.no_grad():
        for i,snapshot in enumerate(test_dataset):
            y_hat, _, _ = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, None, None)
            predictions.append(y_hat)
            targets.append(snapshot.y)
    
    print(f"testing completed in {time.time() - start_time:.2f} seconds.")
    training_time = time.time() - start_time
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    print(f"Number of predictions: {len(targets)}")
mse = torch.mean(torch.abs(predictions - targets)**2)
print("Mean Squared Error:", mse)

# Count the number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")
# with open('predictions_.pkl', 'wb') as f:
#     pickle.dump(predictions, f)

# with open('targets_.pkl', 'wb') as f:
#     pickle.dump(targets, f)