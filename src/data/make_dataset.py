import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pickle
import json



def process_data_and_save():
    # Change the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    nodes = pd.read_csv('../../data/interim/nodes_5.csv')
    lines = pd.read_csv('../../data/interim/lines_5.csv')

    cons_id = 0
    cons_id_node = {}

    pv_id = 0
    pv_id_node = {}

    ev_id = 0
    ev_id_node = {}

    dg_id = 0
    dg_id_node = {}

    ess_id = 0
    ess_id_node = {}
    for row in nodes.iterrows():

        for j in range(len(row[1])):

            if row[1][j] == 'cons':
                cons_id += 1
                cons_id_node[cons_id] = row[1][0]
            
            if row[1][j] == 'pv':
                pv_id += 1
                pv_id_node[pv_id] = row[1][0]
            
            if row[1][j] == 'ev':
                ev_id += 1
                ev_id_node[ev_id] = row[1][0]

            if row[1][j] == 'ess':
                ess_id += 1
                ess_id_node[ess_id] = row[1][0]
            
            if row[1][j] == 'dg':
                dg_id += 1
                dg_id_node[dg_id] = row[1][0]

    with open(r'C:\Users\edi\GitHub\pinns_opf\data\interim\variable_data_big.json') as f:
        dataset = json.load(f)

    hetero_graph_dataset = dict()
    node_id = nodes['Nodes'].values
    for node in node_id:
        hetero_graph_dataset[node] = []
        
    for key in dataset.keys():

        for node_id in dataset[key].keys():
            
            if 'cons' in key:
                node_id_g = cons_id_node[int(node_id)]
                hetero_graph_dataset[node_id_g].append(dataset[key][str(node_id_g)])
            elif 'ev' in key:
                node_id_g = ev_id_node[int(node_id)]
                hetero_graph_dataset[node_id_g].append(dataset[key][str(node_id_g)])
            elif 'pv' in key:
                node_id_g = pv_id_node[int(node_id)]
                hetero_graph_dataset[node_id_g].append(dataset[key][str(node_id)])
            elif 'ess' in key:
                node_id_g = ess_id_node[int(node_id)]
                hetero_graph_dataset[node_id_g].append(dataset[key][str(node_id)])
            elif 'dg' in key:
                node_id_g = dg_id_node[int(node_id)]
                hetero_graph_dataset[node_id_g].append(dataset[key][str(node_id)])
            elif 'V' in key:
                hetero_graph_dataset[int(node_id)].append(dataset[key][str(node_id)])

    list_ = []
    for node,value in hetero_graph_dataset.items():
        hetero_graph_dataset[node] = np.array(value)
        list_.append(np.array(value))

    # Stack the arrays from the dictionary into the result array
    # Create an empty numpy array of size 24x5x9
    result_array = np.empty((24, 5, 13))

    for i, key in enumerate(hetero_graph_dataset.keys()):
        result_array[:, i, :] = hetero_graph_dataset[key].T

    # Reshape the array to (120, 9) to apply normalization to each feature
    data_reshaped = np.reshape(result_array, (result_array.shape[0]*result_array.shape[1], result_array.shape[2]))

    # Calculate the mean and standard deviation along the 0th axis (rows)
    mean = np.mean(data_reshaped, axis=0)
    std = np.std(data_reshaped, axis=0)

    # Normalize the data using mean and standard deviation
    normalized_data = (data_reshaped - mean) / std

    # Reshape the normalized data back to the original shape
    normalized_data = np.reshape(normalized_data, (result_array.shape[0], result_array.shape[1], result_array.shape[2]))
    normalized_data[:,:,-1] = result_array[:,:,-1]

    edge_index = np.array([lines['From'].values,lines['To'].values])
    edge_weigth = np.ones(edge_index.shape[1])


    features = normalized_data[:-1,:,:]
    targets = normalized_data[1:,:,:]


    features_seq = []
    targets_seq = []

    for i in range(1000*len(features[:,1,1])):
        features_seq.append(features[i%23,:,:])
        targets_seq.append(targets[i%23,:,:])



    # Create a dictionary to hold the data
    data = {
        'edge_index': edge_index,
        'edge_weight': edge_weigth,
        'features_seq': features_seq,
        'targets_seq': targets_seq
    }

    # Save the data as a pickle file
    with open('../../data/processed/data_big.pickle', 'wb') as file:
        pickle.dump(data, file)

    
if __name__ == "__main__":

    process_data_and_save()
