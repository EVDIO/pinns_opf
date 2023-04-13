from pyomo.environ import *
from data_opf import get_data_time,get_data_network,get_data_dg,get_data_pv,get_data_cons,get_data_ess


def param_var_time(model,path_time):

    """
    Sets the time slot data for the optimization model.

    Args:
        model: Pyomo abstract model object
        path_time (str): Path to the time slot data file

    Returns:
        Pyomo abstract model object with time slot data set as parameters
    """

    # Get time slot data
    OT,sc_res,Delta_t,pt = get_data_time(path_time)

    # Set time nodes
    model.OT = Set(initialize=OT)

    # Parameters
    model.Delta_t = Param(model.OT, initialize=Delta_t, mutable=True)  # time interval
    model.pt = Param(model.OT, initialize=pt, mutable=True)  # wholesale market price
    
    return model

def param_var_dg(model,path_dg):

    """
    Sets the generator data for the optimization model.

    Args:
        model: Pyomo abstract model object
        path_dg (str): Path to the generator data file

    Returns:
        Pyomo abstract model object with generator data set as parameters and variables
    """

    # Get generator data
    Odg,DG_Pmin,DG_Pmax,DG_fp_min,DG_ramp_up,DG_ramp_dw,DG_a = get_data_dg(path_dg)

    # Set generator nodes
    model.Odg = Set(initialize=Odg)

    # Parameters
    model.DG_Pmin = Param(model.Odg, initialize=DG_Pmin, mutable=True)  # minimum power DGs
    model.DG_Pmax = Param(model.Odg, initialize=DG_Pmax, mutable=True)  # maximum power DGs
    model.DG_fp_min = Param(model.Odg, initialize=DG_fp_min, mutable=True)  # minimum power factor DGs
    model.DG_ramp_up = Param(model.Odg, initialize=DG_ramp_up, mutable=True)  # ramp up constraint
    model.DG_ramp_dw = Param(model.Odg, initialize=DG_ramp_dw, mutable=True)  # ramp down constraint

    # Variables
    model.dg_x =  Var(model.Odg, model.OT, initialize=1.0, within=NonNegativeReals)    # multiplier demands generators (0,1)
    model.Qdg = Var(model.Odg, model.OT, within=Reals) # reactive power for DGs 
    model.DG_cost = Var(model.Odg, initialize=1.0)    # cost DGs
    model.DG_a = Param(model.Odg, initialize=DG_a, mutable=True)  # quadratic coefficient cost DGs

    return model


def param_var_con(model,path_con,path_time):

    """
    Sets the consumer data for the optimization model.

    Args:
        model: Pyomo abstract model object
        path_con (str): Path to the consumer data file
        path_time (str): Path to the time slot data file

    Returns:
        Pyomo abstract model object with consumer data set as parameters and variables
    """

    # Get consumer data
    Ocons,CONS_nodes,CONS_Pmin,CONS_Pmax,CONS_pf,CONS_an,CONS_bn,PM,QM,PD,QD = get_data_cons(path_con,path_time)

    # Set nodes
    model.Ocons = Set(initialize=Ocons)

    # Parameters
    model.CONS_nodes = Param(model.Ocons, initialize=CONS_nodes, mutable=False) # Reactive power bus demand
    model.CONS_Pmin = Param(model.Ocons, initialize=CONS_Pmin, mutable=True)  # Minimum power CONS
    model.CONS_Pmax = Param(model.Ocons, initialize=CONS_Pmax, mutable=True)  # Maximum power CONS
    model.CONS_pf = Param(model.Ocons, initialize=CONS_pf, mutable=True)  # Constant power factor CONS
    model.PM = Param(model.Ocons,model.OT, initialize=PM, mutable=True) # Active power demand for CONS
    model.CONS_an = Param(model.Ocons, initialize=CONS_an, mutable=True)  # quadratic component cost CONS

    # Variables
    model.cons_x = Var(model.Ocons, model.OT, initialize=1.0, within=NonNegativeReals)    # Multiplier demands CONSUMERS (0,1)
    model.Qcons = Var(model.Ocons, model.OT, within=NonNegativeReals) # Reactive power for CONSUMERS
    model.CONS_cost = Var(model.Ocons, initialize=1.0) 

    return model

def param_var_pv(model,path_pv,path_time):
        
    """
    Sets the pv data for the optimization model.

    Args:
        model: Pyomo abstract model object
        path_con (str): Path to the pv data file
        path_time (str): Path to the time slot data file

    Returns:
        Pyomo abstract model object with pv data set as parameters and variables
    """

    # Get pv data
    Opv,PV_Pmin,PV_Pmax,G,PV_pf,PV_sn = get_data_pv(path_pv,path_time)
    
    model.Opv = Set(initialize=Opv)

    model.PV_Pmax = Param(model.Opv, initialize=PV_Pmax, mutable=True) # Maximum active power PV
    model.PV_Pmin = Param(model.Opv, initialize=PV_Pmin, mutable=True)  # Minimum power PVs
    model.PV_pf = Param(model.Opv, initialize=PV_pf, mutable=True)  # Constant power factor PVs
    model.G = Param(model.Opv, model.OT, initialize=G, mutable=True) # Maximum active power for PVs 
    model.PV_sn = Param(model.Opv, initialize=PV_sn, mutable=True)  # Compensation RES

    model.pv_x = Var(model.Opv, model.OT, initialize=1.0, within=NonNegativeReals)    # Multiplier demands PV (0,1)
    model.Qpv = Var(model.Opv, model.OT, within=Reals)
    model.PV_cost = Var(model.Opv, initialize=1.0)    # Multiplier demands CONSUMERS (0,1)

    return model


def param_var_ess(model,path_ess):

    """
    Sets the energy storage system data for the optimization model.

    Args:
        model: Pyomo abstract model object
        path_dg (str): Path to the energy storage system data file

    Returns:
        Pyomo abstract model object with energy storage system data set as parameters and variables
    """

    # Get ESS data
    Oess,ESS_Pmin,ESS_Pmax,ESS_pf_min,ESS_EC,ESS_SOC_ini,ESS_SOC_end,ESS_dn,ESS_SOC_min,ESS_SOC_max = get_data_ess(path_ess)

    # Parameters
    model.ESS_Pmin = Param(model.Oess, initialize=ESS_Pmin, mutable=True)  # Minimum power ESS
    model.ESS_Pmax = Param(model.Oess, initialize=ESS_Pmax, mutable=True)  # Maximum power ESS
    model.ESS_pf_min = Param(model.Oess, initialize=ESS_pf_min, mutable=True)  # Minimum power factor ESS
    model.ESS_SOC_end = Param(model.Oess, initialize=ESS_SOC_end, mutable=True)  # Final state of charge ESS
    model.ESS_SOC_ini = Param(model.Oess, initialize=ESS_SOC_ini, mutable=True)  # Initial state of charge ESS
    model.ESS_SOC_min = Param(model.Oess, initialize=ESS_SOC_min, mutable=True)  # Minimum state of charge ESS
    model.ESS_SOC_max = Param(model.Oess, initialize=ESS_SOC_max, mutable=True)  # MAximum state of charge ESS
    model.ESS_dn = Param(model.Oess, initialize=ESS_dn, mutable=True)  # Depretiation cost ESS
    model.ESS_EC = Param(model.Oess, initialize=ESS_EC, mutable=True)  # Energy capacity ESS
    
    # Variables 
    model.ess_x = Var(model.Oess, model.OT, initialize=1.0, within=NonNegativeReals)    # Multiplier demands PV (0,1)
    model.Qess = Var(model.Oess,model.OT, initialize=0.0) 
    model.SOC = Var(model.Oess,model.OT, initialize=1.0)    # Reactive power from ESS
    model.ESS_cost = Var(model.Oess, initialize=1.0)    # Reactive power from ESS

    return model




def get_param_var_ev(model):
    return model