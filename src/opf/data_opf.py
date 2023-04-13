# Import libraries
import pandas as pd


def get_data_time(path_time):

    """
    Read time data from csv file and return relevant information as dictionaries and lists.
    :param path_time: str - path to the csv file containing time data
    :return: tuple - a tuple containing lists and dictionaries of relevant time data
    """

    # read csv to dataframe
    df_time= pd.read_csv(path_time) 
    
    # time slot data
    OT = [df_time.loc[i, 'T'] for i in df_time.index] # list of time points
    sc_res = {OT[i]: df_time.loc[i, 'Irrad'] for i in df_time.index} 
    Delta_t = {OT[i]: df_time.loc[i, 'Delta_t'] for i in df_time.index} # time step 
    pt = {OT[i]: df_time.loc[i, 'pt'] for i in df_time.index} # market price of [kw/h]

    return [OT,sc_res,Delta_t,pt]

def get_data_network(path_bus,path_line):

    """
    Read network data from csv files and return relevant information as dictionaries and lists.
    :param path_bus: str - path to the csv file containing bus data
    :param path_line: str - path to the csv file containing line data
    :return: tuple - a tuple containing lists and dictionaries of relevant network data
    """

    # read csv to dataframe
    df_bus = pd.read_csv(path_bus) # bus data
    df_line= pd.read_csv(path_line) # line data

    # Network Data
    # Buses
    Ob = [df_bus.loc[i,'name'] for i in df_bus.index] # bus indexes
    Vmin = [df_bus.loc[i,'min_vm_pu'] for i in df_bus.index] # minimum volt [pu]
    Vmax = [df_bus.loc[i,'max_vm_pu'] for i in df_bus.index] # maximum volt [pu]

    # Branches
    Ol = {(df_line.loc[i, 'from_bus'], df_line.loc[i, 'to_bus']) for i in df_line.index}  # nodes of lines
    R = {(df_line.loc[i, 'from_bus'], df_line.loc[i, 'to_bus']): df_line.loc[i, 'r_ohm_per_km']  for i in df_line.index}  # resistance of lines [ohm/km]  note: line is 1km                             # Branch PVistance in pu
    X = {(df_line.loc[i, 'from_bus'], df_line.loc[i, 'to_bus']): df_line.loc[i, 'x_ohm_per_km'] for i in df_line.index}   # reactance of line [ohm/km]
    Imax = {(df_line.loc[i, 'from_bus'], df_line.loc[i, 'to_bus']): df_line.loc[i, 'max_i_ka'] for i in df_line.index}  # maximum current [kA]

    Ol = sorted(Ol, key=lambda x: x[0])

    return [Ob,Ol,Vmin,Vmax,R,X,Imax]

def get_data_dg(path_dg):

    """
    Read data about distributed generators (DGs) from csv file and return relevant information as dictionaries and lists.
    :param path_dg: str - path to the csv file containing DG data
    :return: tuple - a tuple containing lists and dictionaries of relevant DG data
    """

    # read csv to dataframe
    df_gen = pd.read_csv(path_dg)

    # Generators - DGs
    Odg = [df_gen.loc[i, 'bus'] for i in df_gen.index]      # nodes with DG
    DG_Pmin = {Odg[i]: df_gen.loc[i, 'min_p_mw'] for i in df_gen.index}     # minimum active power DG [mW]
    DG_Pmax = {Odg[i]: df_gen.loc[i, 'max_p_mw'] for i in df_gen.index}     # maximum active power DG [mW]
    DG_fp_min = {Odg[i]: df_gen.loc[i, 'fp_min'] for i in df_gen.index}     # minimum power factor DG
    DG_ramp_up = {Odg[i]: df_gen.loc[i, 'ramp_up'] for i in df_gen.index}   # ramp up constraint
    DG_ramp_dw = {Odg[i]: df_gen.loc[i, 'ramp_down'] for i in df_gen.index} # ramp down constraint
    DG_a = {Odg[i]: df_gen.loc[i, 'DG_a'] for i in df_gen.index}            # quadratic coefficient cost DG

    return [Odg,DG_Pmin,DG_Pmax,DG_fp_min,DG_ramp_up,DG_ramp_dw,DG_a]

def get_data_pv(path_pv,path_time):

    """
    Read data about photovoltaics (PVs) and time data from csv file and return relevant information as dictionaries and lists.
    :param path_pv: str - path to the csv file containing pv data
    :param path_time: str - path to the csv file containing time data
    :return: tuple - a tuple containing lists and dictionaries of relevant pv data
    """

    # read csv to dataframe
    df_pv  = pd.read_csv(path_pv)
    df_time = pd.read_csv(path_time)

    # PV
    Opv = [df_pv.loc[i, 'PV_node'] for i in df_pv.index]             # nodes with PV
    PV_pf = {Opv[i]: df_pv.loc[i, 'PV_pf'] for i in df_pv.index}     # power factor PV
    PV_sn = {Opv[i]: df_pv.loc[i, 'PV_sn'] for i in df_pv.index}     # per unit subsidy support for PVs
    PV_Pmin = {Opv[i]: df_pv.loc[i, 'PV_min'] for i in df_pv.index}  # minimum power PV [mW]
    PV_Pmax = {Opv[i]: df_pv.loc[i, 'PV_max'] for i in df_pv.index}  # maximum power PV [mW]
    OT = [df_time.loc[i, 'T'] for i in df_time.index]                # list of time points
    G =  {(Opv[i], OT[t]): df_time['PV_{}_{}'.format(df_pv['PV_node'][i],i)][t] for i in df_pv.index for t in df_time.index} # maximum power PV for each hour

    return [Opv,PV_Pmin,PV_Pmax,G,PV_pf,PV_sn]

def get_data_cons(path_con,path_time):

    """
    Read data about consumers and time data from csv file and return relevant information as dictionaries and lists.
    :param path_con: str - path to the csv file containing consumer data
    :param path_time: str - path to the csv file containing time data
    :return: tuple - a tuple containing lists and dictionaries of relevant pv data
    """

    # read csv to dataframe
    df_con  = pd.read_csv(path_con)
    df_time = pd.read_csv(path_time)

    # consumers 
    Ocons = [i+1 for i in df_con.index]                                       # index of consumers
    CONS_nodes = {Ocons[i]: df_con.loc[i, 'Node_cons'] for i in df_con.index} # nodes with consumers
    CONS_Pmin = {Ocons[i]: df_con.loc[i, 'CONS_Pmin'] for i in df_con.index}  # minimum power consumers [mw]
    CONS_Pmax = {Ocons[i]: df_con.loc[i, 'CONS_Pmax'] for i in df_con.index}  # maximum power consumers [mw]
    CONS_an = {Ocons[i]: df_con.loc[i, 'CONS_an'] for i in df_con.index}      # instantaneous cost load shifted
    CONS_bn = {Ocons[i]: df_con.loc[i, 'CONS_bn'] for i in df_con.index}      # extra cost load unsatisfied
    PD = {Ocons[i]: df_con.loc[i, 'PDn'] for i in df_con.index}               # nominal nodal active demand in pu
    QD = {Ocons[i]: df_con.loc[i, 'QDn']  for i in df_con.index}              # nominal nodal reactive demand in pu
    CONS_pf = {Ocons[i]: df_con.loc[i, 'CONS_pf'] for i in df_con.index}      # minimum power factor for consumers
    
    # Demand for consumers
    OT = [df_time.loc[i, 'T'] for i in df_time.index] # list of time points
    PM = {(Ocons[i], OT[t]): df_time['PD_{}_{}'.format(df_con['Node_cons'][i],i)][t] for i in df_con.index for t in df_time.index} # active power demand [mW]
    QM = {(Ocons[i], OT[t]): df_time['QD_{}_{}'.format(df_con['Node_cons'][i],i)][t] for i in df_con.index for t in df_time.index} # reactive power demand [mW]

    return [Ocons,CONS_nodes,CONS_Pmin,CONS_Pmax,CONS_pf,CONS_an,CONS_bn,PM,QM,PD,QD]

def get_data_ess(path_ess):

    """
    Read data about energy storages (ESS) from csv file and return relevant information as dictionaries and lists.
    :param path_ess: str - path to the csv file containing ess data
    :return: tuple - a tuple containing lists and dictionaries of relevant ESS data
    """

    # read csv to dataframe
    df_ess = pd.read_csv(path_ess)

    # Energy storage systems (ESS)
    Oess = [df_ess.loc[i, 'ESS_node'] for i in df_ess.index]                              # nodes with ESS
    ESS_Pmin = {Oess[i]: df_ess.loc[i, 'ESS_Pmin'] for i in df_ess.index}                 # minimum power ESS
    ESS_Pmax = {Oess[i]: df_ess.loc[i, 'ESS_Pmax'] for i in df_ess.index}                 # maximum power ESS
    ESS_pf_min = {Oess[i]: df_ess.loc[i, 'ESS_pf_min'] for i in df_ess.index}             # minimum power factor ESS
    ESS_EC = {Oess[i]: df_ess.loc[i, 'ESS_EC'] for i in df_ess.index}                     # energy capacity ESS
    ESS_SOC_ini = {Oess[i]: df_ess.loc[i, 'ESS_SOC_ini']/100 for i in df_ess.index}       # initial state of charge ESS
    ESS_SOC_end = {Oess[i]: df_ess.loc[i, 'ESS_SOC_end']/100 for i in df_ess.index}       # end state of charge ESS
    ESS_dn = {Oess[i]: df_ess.loc[i, 'ESS_dn'] for i in df_ess.index}                     # battery  degradation cost ESS
    ESS_SOC_min = {Oess[i]: df_ess.loc[i, 'SOC_min']/100 for i in df_ess.index}           # minimum state of charge ESS
    ESS_SOC_max = {Oess[i]: df_ess.loc[i, 'SOC_max']/100 for i in df_ess.index}           # maximum state of charge ESS

    return [Oess,ESS_Pmin,ESS_Pmax,ESS_pf_min,ESS_EC,ESS_SOC_ini,ESS_SOC_end,ESS_dn,ESS_SOC_min,ESS_SOC_max]


######### TO DO #################################3


# # Electric vehicles (EV)
# Oev = [i+1 for i in df_ev.index]                                      # Index EVs
# EV_nodes = {i:df_ev.loc[i-1, 'EV_node'] for i in Oev}                                      # Nodes with EVs
# EV_Pmin = {i:df_ev.loc[i-1, 'EV_Pmin'] for i in Oev}                                  # Minimum power EVs
# EV_Pmax = {i:df_ev.loc[i-1, 'EV_Pmax'] for i in Oev}                                # Maximum power EVs
# EV_Pnom = {i:df_ev.loc[i-1, 'EV_Pnom']/ (Snom) for i in Oev}                               # Nominal power EVs
# EV_pf_min = {i:df_ev.loc[i-1, 'EV_pf_min'] for i in Oev}                              # Minimum power factor EVs
# EV_EC = {i:df_ev.loc[i-1, 'EV_EC']/ (Snom) for i in Oev}                                      # Energy capacity EVs
# EV_SOC_ini = {i:df_ev.loc[i-1, 'EV_SOC_ini']/100 for i in Oev}                            # Arrive state of charge
# EV_SOC_end = {i:df_ev.loc[i-1, 'EV_SOC_end']/100 for i in Oev}                            # Departure state of charge
# EV_SOC_min = {i: df_ev.loc[i-1, 'EV_SOC_min']/100 for i in Oev}                        # Minimum state of charge ESS
# EV_SOC_max = {i: df_ev.loc[i-1, 'EV_SOC_max']/100 for i in Oev}                        # Maximum state of charge ESS
# EV_dn = {i:df_ev.loc[i-1, 'EV_dn'] for i in Oev}                                      # Battery  degradation cost
# EV_wn = {i:df_ev.loc[i-1, 'EV_wn'] for i in Oev}                                      # Penalty  cost  for  not  having  their  battery  charged  at  the desired level
# t_arr = {i:df_ev.loc[i-1, 't_arr'] for i in Oev}                                      # Time of arrival
# t_dep = {i:df_ev.loc[i-1, 't_dep'] for i in Oev}                                      # Time of departure
