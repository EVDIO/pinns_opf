import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import pickle
import matplotlib.pyplot as plt
from model import RecurrentGCN


def load_data():
    # Load the data from the pickle file
    with open(r'C:\Users\edi\GitHub\pinns_opf\data\processed\data_converged_noise2.pickle', 'rb') as file:
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

def main():
    # Load the test dataset
    test_dataset = load_data()

    # Load the saved model
    model_path = r"C:\Users\edi\GitHub\pinns_opf\src\models\model_20230911174441.pt"  # replace with your model path or provide a way to input it
    model = RecurrentGCN(node_features=9)
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    predictions, targets = evaluate_model(model, test_dataset)


if __name__ == "__main__":
    main()
