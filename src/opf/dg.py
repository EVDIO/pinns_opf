# Import libraries
import pandas as pd
from pyomo.environ import *



# Define paths
pth = r'C:\Users\Dell\Documents\GitHub\PINNS_OPF\data\interim'
gen = r'\asset_gen.csv'
# Load data
df_gen = pd.read_csv(pth+gen)
