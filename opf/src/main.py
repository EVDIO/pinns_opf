from pyomo.environ import *
import json
import numpy as np

from model.opf import run_opf

import numpy as np
# Initiate model
model = ConcreteModel()


model,result = run_opf(model)


#model.display()


# for parmobject in model.component_objects(Param, active=True):
#     nametoprint = str(str(parmobject.name))
#     print ("Parameter ", nametoprint)  
#     for index in parmobject:
#         vtoprint = value(parmobject[index])
#         print ("   ",index, vtoprint)  


for v in model.component_objects(Var, active=True):
    print("Variable",v)
    for index in v:
        print ("   ",index, value(v[index])) 

variable_data = {}
for v in model.component_objects(Var, active=True):
    variable_name = v.name
    
    if variable_name in ['dg_x', 'Qdg', 'cons_x', 'Qcons', 'ess_x', 'Qees', 'pv_x', 'Qpv']:
        variable_data[variable_name] = {}
        
        for i,index in enumerate(v):
            key = int(index[0])
            values = [value(v[index[0], t]) for t in range(1, 25)]
            variable_data[variable_name][key] = values
            key_prev = key

list_data = []
for key in variable_data.keys():
    for s_key in variable_data[key].keys():
        list_data.append(variable_data[key][s_key])

data_array = np.array(list_data,ndmin=2).T

with open('data.npy', 'wb') as f:
    np.save(f, data_array)
    
# Save variable_data to a JSON file
with open('variable_data.json', 'w') as json_file:
    json.dump(variable_data, json_file)
