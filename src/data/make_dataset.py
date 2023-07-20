# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import json
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import pickle



def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """


    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    nodes = pd.read_csv(r'C:\Users\Dell\Documents\GitHub\pinns_opf\data\interim\nodes_33.csv')

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


    dg_id_node = np.array(list(dg_id_node.items())).T
    ess_id_node = np.array(list(ess_id_node.items())).T
    pv_id_node = np.array(list(pv_id_node.items())).T
    ev_id_node = np.array(list(ev_id_node.items())).T
    cons_id_node = np.array(list(cons_id_node.items())).T
    node_cons_id = np.array((cons_id_node[1],cons_id_node[0]))
    with open(r'C:\Users\Dell\Documents\GitHub\pinns_opf\notebooks\variable_data.json') as f:
        dataset = json.load(f)

    def create_sequence_ders(dataset, key1, key2, key3):
        # Assuming you have two NumPy arrays
        data_key1 = np.array(list(dataset[key1].values())).T
        data_key2 = np.array(list(dataset[key2].values())).T

        # Create a sequence of dictionaries
        sequence = []

        for i in range(data_key1.shape[0]):
            # Stack and transpose the elements
            stacked_elements = np.vstack((data_key1[i], data_key2[i])).T

            # Create a dictionary with 'dg' as the key and stacked_elements as the value
            dictionary = {key3 : stacked_elements}

            # Append the dictionary to the sequence
            sequence.append(dictionary)
        
        return sequence

    def create_sequence_nodes(dataset, key1, key2):
        # Assuming you have two NumPy arrays
        data_key1 = np.array(list(dataset[key1].values())).T

        # Create a sequence of dictionaries
        sequence = []

        for i in range(data_key1.shape[0]):
            # Stack and transpose the elements
            elements = np.array(data_key1[i]).T

            elements = np.reshape(elements, (-1, 1))
            # Create a dictionary with 'dg' as the key and stacked_elements as the value
            dictionary = {key2 : elements}
            # Append the dictionary to the sequence
            sequence.append(dictionary)
        
        return sequence

    # Example usage
    sequence_dg = create_sequence_ders(dataset, 'dg_x', 'Qdg','dg')
    sequence_cons = create_sequence_ders(dataset, 'cons_x', 'Qcons','cons')
    sequence_ess = create_sequence_ders(dataset, 'ess_x', 'Qess', 'ess')
    sequence_ev = create_sequence_ders(dataset, 'Pev', 'Qev', 'ev')
    sequence_pv = create_sequence_ders(dataset, 'pv_x', 'Qpv', 'pv')
    sequence_nodes = create_sequence_nodes(dataset, 'V', 'nodes')


    combined_sequence = []

    for dict1, dict2 in zip(sequence_cons, sequence_nodes):
        combined_dict = {**dict1, **dict2}  # Merge the dictionaries
        combined_sequence.append(combined_dict)

    edge_index_dict = {
    ("cons","connect","nodes") : cons_id_node,
    ("nodes","connect","cons") : node_cons_id
    }

    feature_dicts = combined_sequence[:-1]
    target_dicts = combined_sequence[1:]

    # Define the file path to save the dictionaries
    file_path = r'C:\Users\Dell\Documents\GitHub\pinns_opf\data\processed\dictionaries.pkl'

    # Create a dictionary to hold all the required data
    data_dict = {
        'edge_index_dict': edge_index_dict,
        'feature_dicts': feature_dicts,
        'target_dicts': target_dicts
    }

    # Save the dictionary to a file using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(data_dict, file)

    print("Dictionaries saved successfully.")

    data = HeteroData()
    data['cons'].x = sequence_cons
    data['nodes'].x = sequence_nodes[:-1]
    data['nodes'].y = sequence_nodes[1:]

    data['cons','connects','nodes'].edge_index = cons_id_node
    metadata = data.metadata()
        # Define the file path to save the metadata
    file_path = r'C:\Users\Dell\Documents\GitHub\pinns_opf\data\processed\metadata.pkl'

    # Save the metadata tuple to a file using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(metadata, file)

    print("Metadata saved successfully.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()
