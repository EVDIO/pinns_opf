import torch
import torch.nn.functional as F
#from torch_geometric_temporal.nn.recurrent import DCRNN,GCLSTM
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from tqdm import tqdm
import datetime


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvLSTM(node_features, 32, 1, normalization='rw')
        self.linear = torch.nn.Linear(32, 9)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
        

    def _train(self, model, train_dataset, epochs, lr=0.01, h=None, c=None, pinns_loss=None, _lambda = None, v_lower=None, v_upper=None):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        costs = []

        for epoch in range(epochs):
            cost = 0
            h=None; c=None
            # Training loop over the dataset
            for time, snapshot in enumerate(train_dataset):
                y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
                v_pred = y_hat[:,-1]
                if _lambda is None:
                    cost = cost + torch.mean((y_hat - snapshot.y) ** 2) 
                else:
                    cost = cost + torch.mean((y_hat - snapshot.y) ** 2) + _lambda*pinns_loss(v_pred, v_lower, v_upper)

            cost = cost / (time+1)
            costs.append(float(cost))
            cost.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch+1}/{epochs} - Cost: {cost:.4f}")

        # Save the model at the end of training
        model_path = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
        torch.save(model.state_dict(), model_path)

        return costs, model_path
    
    def evaluate(model, eval_dataset, h=None, c=None):
        model.eval()
        predictions = []
        targets = []

        
        for time,snapshot in enumerate(eval_dataset):
            y_hat, _, _ = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
            predictions.append(y_hat)
            targets.append(snapshot.y)

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        return predictions, targets

