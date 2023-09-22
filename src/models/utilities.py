import torch
import torch.nn.functional as F

def voltage_loss(v_pred, v_lower, v_upper):
    """
    Computes the loss for inequality constraints v_lower <= v_pred <= v_upper.
    
    Arguments:
    v_pred -- predicted vector of size 5x1
    v_lower -- lower bound vector of size 5x1
    v_upper -- upper bound vector of size 5x1
    
    Returns:
    loss -- the computed loss
    """
    # Compute violation of lower bound
    violation_lower = torch.relu(v_lower - v_pred)
    
    # Compute violation of upper bound
    violation_upper = torch.relu(v_pred - v_upper)
    
    # Compute the total violation
    total_violation = torch.sum(violation_lower) + torch.sum(violation_upper)
    
    # Compute the hinge loss
    hinge_loss = F.relu(total_violation)
    
    return hinge_loss



def powerflow_loss(y_pred):
    # Assuming the ordering in y_pred is P, Q, V, I
    
    P_b = y_pred[:, 0:5]
    Q_b = y_pred[:,5:10]
    V = y_pred[:, -2]
    I = y_pred[:, -1]
    P_flow, Q_flow = get_powerFlows(P_b, Q_b, I)
    # Calculate the SOCP-based regularization term
    socp_term = V * I - P_flow**2 - Q_flow**2
    
    loss = torch.mean(F.relu(socp_term))

    return loss

def get_powerFlows(P_b, Q_b, I):

    network_data = {
        (1, 2): {'R': 0.117, 'X': 0.048},
        (2, 3): {'R': 0.1073, 'X': 0.044},
        (3, 4): {'R': 0.1645, 'X': 0.0457},
        (3, 7): {'R': 0.1572, 'X': 0.027},
        (4, 5): {'R': 0.1495, 'X': 0.0415},
        (5, 6): {'R': 0.1495, 'X': 0.0415},
        (6, 9): {'R': 0.1794, 'X': 0.0498},
        (9, 10): {'R': 0.1645, 'X': 0.0457},
        (7, 8): {'R': 0.2096, 'X': 0.036}
    }

    P_flow = torch.zeros(10)
    Q_flow = torch.zeros(10)

    for b in range(1, 11):  # buses from 1 to 10
        # Nodes sum
        P_nodes_sum = torch.sum(P_b[b-1, :])
        Q_nodes_sum = torch.sum(Q_b[b-1, :])

        # Branches sum
        P_branch_sum = 0
        Q_branch_sum = 0

        # Iterate through each connection for the bus 'b'
        for key, values in network_data.items():
            from_node, to_node = key
            if from_node == b or to_node == b:  # If the branch is connected to bus 'b'
                R = values['R']
                X = values['X']
                
                branch_index = from_node if from_node != b else to_node
                P_branch_sum += P_b[branch_index - 1, :].sum() + R * (I[branch_index-1]**2)
                Q_branch_sum += Q_b[branch_index - 1, :].sum() + X * (I[branch_index-1]**2)

        # Update the P_flow and Q_flow for the bus 'b'
        P_flow[b-1] = P_nodes_sum - P_branch_sum
        Q_flow[b-1] = Q_nodes_sum - Q_branch_sum

    return P_flow, Q_flow

