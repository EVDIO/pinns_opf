"""
This script is simulating the AC-OPF for DERs using pyomo library

"""
# Import libraries
from pyomo.environ import *
from param_var_opf import *

# Define paths
pth = r'C:\Users\Dell\Documents\GitHub\PINNS_OPF\data\interim'
con = r'\asset_cons.csv'
ess = r'\asset_ess.csv'
ev = r'\asset_ev.csv'
dg = r'\asset_gen.csv'
pv  = r'\asset_pv.csv'
bus = r'\bus5.csv'
line= r'\line5.csv'
time= r'\time_slots.csv'

# Initiate model
model = ConcreteModel()

# Set variables and parameters for time data
model = param_var_time(model=model, path_time=pth+time)
# Set variables and parameters for dg data
model = param_var_dg(model=model, path_dg=pth+dg)
# Set variables and parameters for comsumers
model = param_var_con(model=model, path_con=pth+con,
                      path_time=pth+time)
# Set variables and parameters for pv
model = param_var_pv(model=model, path_pv=pth+pv,
                     path_time=pth+time)
# Set variables and parameters for ess
model = param_var_ess(model=model, path_ess=pth+ess)
# Set variables and parameters for ev
model = param_var_ev(model=model, path_ev=pth+ev)
# Set variables and parameters for network
model = param_var_network(model=model, path_bus=pth+bus,
                          path_line=pth+line)

# Set rules for DGs
def DG_decision_var_rule(model,i,t):
    """
    decision variable should be bounded (0,1)
    """
    return (0,model.dg_x[i,t],1)
model.DG_decision_var = Constraint(model.Odg, model.OT,
                                   rule = DG_decision_var_rule)

def DG_constr_active_power_rule(model, i, t):
    """
    Rule for active power of generators 
        Pmin <= x*Pmax <= Pmax
    """
    return (model.DG_Pmin[i], model.dg_x[i,t]*model.DG_Pmax[i], model.DG_Pmax[i])
model.DG_constr_active_power = Constraint(model.Odg, model.OT,
                                          rule=DG_constr_active_power_rule)  # Minimum, Maximum power from DG

def DG_constr_reactive_power_rule(model, i, t):
    """
    Rule for reactive power of generators 
         -Q <= x*Pmax*tan(acos(fp_min))
    """
    return ( -model.Qdg[i,t] <= model.dg_x[i,t]*model.DG_Pmax[i]*tan(acos(model.DG_fp_min[i])))
model.DG_constr_reactive_power = Constraint(model.Odg, model.OT,
                                    rule=DG_constr_reactive_power_rule)  # Reactive power for DG

def DG_constr_reactive_power_rule_2(model, i, t):
    """
    Rule for reactive power of generators 
       Q<= x*Pmax*tan(acos(fp_min))
    """
    return (model.Qdg[i,t] <= model.dg_x[i,t]*model.DG_Pmax[i]*tan(acos(model.DG_fp_min[i])))
model.DG_constr_reactive_power_2 = Constraint(model.Odg, model.OT,
                                       rule=DG_constr_reactive_power_rule_2)  # Reactive power for DG
  
def DG_ramp_rule(model, i, t):
    """
    Pn,t - Pn,t-1 <= DG_ramp_upn *Pmax   upper ramping constraint
    Pn,t-1 - Pn,t <= DG_ramp_dwn *Pmax   lower ramping constrain
    """
    if t > 1:
        return (model.DG_ramp_dw[i]*model.DG_Pmax[i], model.dg_x[i,t]*model.DG_Pmax[i] - model.dg_x[i,t-1]*model.DG_Pmax[i] ,
                model.DG_ramp_up[i]*model.DG_Pmax[i])
    else:
        return (model.DG_ramp_dw[i] * model.DG_Pmax[i], model.dg_x[i,t]*model.DG_Pmax[i],
                model.DG_ramp_up[i] * model.DG_Pmax[i])
model.DG_ramp = Constraint(model.Odg,model.OT,
                            rule=DG_ramp_rule)  # ramp-up and ramp-down constrains

# Set rules for consumers
# Question what's should be the maximum and minimum limits 

def CONS_decision_var_rule(model,i,t):
    """
    decision variable should be bounded (0,1)
    """
    return (0,model.cons_x[i,t],1)
model.CONS_decision_var = Constraint(model.Ocons, model.OT,
                                   rule = CONS_decision_var_rule)

def CONS_constr_active_power_rule(model, i, t):
    """
    Rule for active power of consumers 
        Pmin <= x*Pmax <= Pmax
    """
    return (model.CONS_Pmin[i] * model.PM[i,t], model.cons_x[i,t] * model.PM[i,t], model.CONS_Pmax[i] * model.PM[i,t])
model.CONS_constr_active_power = Constraint(model.Ocons, model.OT,
                                    rule=CONS_constr_active_power_rule)  # Minimum, Maximum power for comsumers

def CONS_constr_reactive_power_rule(model, i, t):
    """
    Rule for reactive power of consumers 
       Q == x*Pmax*tan(acos(fp_min))
    """
    return (model.cons_x[i,t] * model.PM[i,t]*tan(acos(model.CONS_pf[i])) == model.Qcons[i,t])
model.CONS_constr_reactive_power = Constraint(model.Oess, model.OT,
                                    rule=CONS_constr_reactive_power_rule)  # Reactive power for consumers


# Set rules for PVs
def PV_decision_var_rule(model,i,t):
    """
    decision variable should be bounded (0,1)
    """
    return (0,model.pv_x[i,t],1)
model.PV_decision_var = Constraint(model.Opv, model.OT,
                                   rule = PV_decision_var_rule)

def PV_constr_active_power_rule(model, i, t):
    """
    Rule for active power of OV 
        Pmin <= x*G <= G
        G is the output of the power generation 
    """
    return (model.PV_Pmin[i] , model.pv_x[i,t] * model.G[i,t] , model.G[i,t])
model.PV_constr_active_power = Constraint(model.Opv, model.OT,
                                          rule=PV_constr_active_power_rule)  # Minimum, Maximum power for PV

def PV_constr_reactive_power_rule(model, i, t):
    """
    Rule for reactive power of consumers 
       Q == x*G*tan(acos(fp_min))
    """
    return (model.pv_x[i,t] *model.G[i,t]*tan(acos(model.PV_pf[i])) == model.Qpv[i,t])
model.PV_constr_reactive_power = Constraint(model.Opv, model.OT,
                                    rule=PV_constr_reactive_power_rule)  # Reactive power from PV


#####################
# ----- ESS -------- #
def ESS_decision_var_rule(model,i,t):
    """
    decision variable should be bounded (0,1)
    """
    return (0,model.ess_x[i,t],1)
model.ESS_decision_var = Constraint(model.Oess, model.OT,
                                   rule = ESS_decision_var_rule)


def ESS_constr_active_power_rule(model, i, t):

    """
    Rule for active power of consumers 
        Pmin <= x*Pmax <= Pmax
    """
    return (model.ESS_Pmin[i], model.ess_x[i,t]*model.ESS_Pmax[i], model.ESS_Pmax[i])
model.ESS_constr_active_power = Constraint(model.Oess, model.OT,
                                    rule=ESS_constr_active_power_rule)  # Minimum, Maximum power from ESS

def ESS_constr_reactive_power_rule(model, i, t):
    
    """
    Rule for reactive power of ess 
      - Q<= x*Pmax*tan(acos(fp_min))
    """
    return (-model.Qess[i,t] <= model.ess_x[i,t]*model.ESS_Pmax[i]*tan(acos(model.ESS_pf_min[i])))
model.ESS_constr_reactive_power = Constraint(model.Oess, model.OT,
                                    rule=ESS_constr_reactive_power_rule)  # Minimum, Maximum reactive power from ESS

def ESS_constr_reactive_power_rule_2(model, i, t):
    """
    Rule for reactive power of ess 
       Q<= x*Pmax*tan(acos(fp_min))
    """
    return (model.Qess[i,t] <= model.ess_x[i,t]*model.ESS_Pmax[i]**tan(acos(model.ESS_pf_min[i])))
model.ESS_constr_reactive_power_2 = Constraint(model.Oess, model.OT,
                                    rule=ESS_constr_reactive_power_rule_2)  # Minimum, Maximum reactive power from ESS

def ESS_constr_SOC_1_rule(model, i, t):
    if t == 1:
        return (model.SOC[i,t] == model.ESS_SOC_ini[i] - model.Delta_t[t]/model.ESS_EC[i]*model.ess_x[i,t]*model.ESS_Pmax[i])
    else:
        return (model.SOC[i,t] == model.SOC[i,t-1] - model.Delta_t[t]/model.ESS_EC[i]*model.ess_x[i,t]*model.ESS_Pmax[i])
model.ESS_constr_SOC_1 = Constraint(model.Oess, model.OT,
                                    rule=ESS_constr_SOC_1_rule)  # Charging

def ESS_constr_SOC_2_rule(model, i, t):
    return (model.ESS_SOC_min[i], model.SOC[i,t], model.ESS_SOC_max[i])
model.ESS_constr_SOC_2 = Constraint(model.Oess, model.OT,
                                    rule=ESS_constr_SOC_2_rule)  # Minimum, Maximum capacity of storage

def ESS_constr_SOC_departure_rule(model, i):
    return (model.ESS_SOC_ini[i] - sum(model.Delta_t[t]/model.ESS_EC[i]*model.ess_x[i,t]*model.ESS_Pmax[i] for t in model.OT) >= model.ESS_SOC_end[i])
model.ESS_constr_SOC_departure = Constraint(model.Oess,
                                    rule=ESS_constr_SOC_departure_rule)  # Minimum, Maximum reactive power from ESS

#####################
# ----- EVs -------- #
def EV_constr_active_power_rule(model, i, t):
    if model.t_arr[i].value <= t <= model.t_dep[i].value:
        return (model.EV_Pmin[i]*model.EV_Pnom[i], model.Pev[i,t], model.EV_Pmax[i]*model.EV_Pnom[i])
    else:
        return (model.Pev[i, t] == 0.0)
model.EV_constr_active_power = Constraint(model.Oev, model.OT,
                                    rule=EV_constr_active_power_rule)  # Minimum, Maximum power from ESS

def EV_constr_reactive_power_rule(model, i, t):
    return (-model.Pev[i,t]*tan(acos(model.EV_pf_min[i])) <= model.Qev[i,t])
model.EV_constr_reactive_power = Constraint(model.Oev, model.OT,
                                    rule=EV_constr_reactive_power_rule)  # Minimum, Maximum reactive power from ESS

def EV_constr_reactive_power_rule_2(model, i, t):
    return (model.Qev[i,t] <= model.Pev[i,t]*tan(acos(model.EV_pf_min[i])))
model.EV_constr_reactive_power_2 = Constraint(model.Oev, model.OT,
                                    rule=EV_constr_reactive_power_rule_2)  # Minimum, Maximum reactive power from ESS
#
#
def EV_constr_SOC_1_rule(model, i, t):
    if t <= model.t_arr[i].value:
        return (model.SOC_EV[i,t] == model.EV_SOC_ini[i] - model.Delta_t[t]/model.EV_EC[i]*model.Pev[i,t])
    else:
        return (model.SOC_EV[i,t] == model.SOC_EV[i,t-1] - model.Delta_t[t]/model.EV_EC[i]*model.Pev[i,t])
model.EV_constr_SOC_1 = Constraint(model.Oev, model.OT,
                                    rule=EV_constr_SOC_1_rule)  # Minimum, Maximum reactive power from ESS

def EV_constr_SOC_2_rule(model, i, t):
    return (model.EV_SOC_min[i], model.SOC_EV[i,t], model.EV_SOC_max[i])
model.EV_constr_SOC_2 = Constraint(model.Oev, model.OT,
                                    rule=EV_constr_SOC_2_rule)  # Minimum, Maximum reactive power from ESS

def EV_constr_SOC_departure_rule(model, i):
    return (model.SOC_EV[i,model.t_dep[i].value] + model.E_non_served[i] >= model.EV_SOC_end[i]) #
model.EV_constr_SOC_departure = Constraint(model.Oev,
                                    rule=EV_constr_SOC_departure_rule)  # Minimum, Maximum reactive power from ESS


#### Cost functions ###########

# Generators
def DG_cost_rule(model, i):
    return (model.DG_cost[i] == sum(model.DG_a[i] *  (model.dg_x[i,t]*model.DG_Pmax[i]) ** 2 +  model.pt[t] *  model.dg_x[i,t]*model.DG_Pmax[i]  for t in model.OT)
)
model.DG_cf = Constraint(model.Odg,
                         rule=DG_cost_rule)  

# Consumers
def CONS_cost_rule(model, i):
    return (model.CONS_cost[i] == sum(sum(model.pt[t]*model.cons_x[i, t]*model.PM[i, t] + model.CONS_an[i] * (model.PM[i, t]) ** 2 * (1 - model.cons_x[i, t]) ** 2 for t in model.OT) for i in model.Ocons)) 
model.CONS_cf = Constraint(model.Ocons,
                           rule=CONS_cost_rule)  

# PV 
def PV_cost_rule(model, i):
    return (model.PV_cost[i] ==  sum((model.pt[t] + model.PV_sn[i])*model.pv_x[i, t]*model.G[i, t] for t in model.OT))
model.PV_cf = Constraint(model.Opv,
                         rule=PV_cost_rule)  

# ESS
def ESS_cost_rule(model, i):
    return (model.ESS_cost[i] == sum(-model.ESS_dn[i]*(model.ess_x[i,t]*model.ESS_Pmax[i])/model.ESS_EC[i] - model.pt[t]*model.ess_x[i,t]*model.ESS_Pmax[i] for t in model.OT))
model.ESS_cf = Constraint(model.Oess,
                          rule=ESS_cost_rule)  # Minimum, Maximum reactive power from ESS

# EV
def EV_cost_rule(model, i):
    return (model.EV_cost[i] == sum(-model.EV_dn[i]*(model.Pev[i,t])/model.EV_EC[i] - model.pt[t]*model.Pev[i,t] for t in model.OT)
            + model.EV_wn[i]* (model.E_non_served[i]))
model.EV_cf = Constraint(model.Oev,
                                    rule=EV_cost_rule)  # Minimum, Maximum reactive power from ESS


# Constrains of the network
def active_power_flow_rule(model, k,t):
        
        active_power_parent = sum(model.P[j,i,t] for j,i in model.Ol if i == k )
        active_power_dg = sum(model.dg_x[i,t]*model.DG_Pmax[i] for i in model.Odg if k == i)
        active_power_pv = sum(model.pv_x[i,t] * model.G[i,t] for i in model.Opv if k==i)
        active_power_ess = sum(model.ess_x[i,t]*model.ESS_Pmax[i] for i in model.Oess if k==i)
        active_power_cons = sum(model.cons_x[i,t] * model.PM[i,t] for i,m in model.CONS_nodes if k==m)
        active_power_ev = sum(model.Pev[i,t] for i in model.Oev for i,m in model.EV_nodes if k==m)
        active_power_child = sum(model.P[i,j,t] + model.R[i,j]*(model.I[i,j,t]) for i,j in model.Ol if k==i)

        return (active_power_parent
                +active_power_dg
                +active_power_pv
                +active_power_ess
                #+active_power_cons
                +active_power_ev
                -active_power_child==0)

model.active_power_flow = Constraint(model.Ob, model.OT, rule=active_power_flow_rule)   # Active power balance

def reactive_power_flow_rule(model, k,t):
        
        reactive_power_parent = sum(model.Q[j,i,t] for j,i in model.Ol if i == k )
        reactive_power_dg     = sum(model.Qdg[i,t] for i in model.Odg if k == i)
        reactive_power_pv     = sum(model.Qpv[i,t] for i in model.Opv if k==i)
        reactive_power_ess    = sum(model.Qess[i,t] for i in model.Oess if k==i)
        reactive_power_cons   = sum(model.Qcons[i,t] for i,m in model.CONS_nodes if k==m)
        reactive_power_pv     = sum(model.Qev[i,t]  for i,m in model.EV_nodes if k==m)
        reactive_power_child  = sum(model.Q[i,j,t] + model.X[i,j]*(model.I[i,j,t]) for i,j in model.Ol if k == i)
        
        return (reactive_power_parent
                +reactive_power_dg
                +reactive_power_pv
                +reactive_power_ess
                #+reactive_power_cons
                +reactive_power_child == 0)

  
# model.reactive_power_flow = Constraint(model.Ob, model.OT, rule=reactive_power_flow_rule) # Reactive power balance

# what is the square voltage here and square current
def voltage_drop_rule(model, i, j,t):
    return (model.V[i,t] - 2*(model.R[i,j]*model.P[i,j,t] + model.X[i,j]*model.Q[i,j,t]) - (model.R[i,j]**2 + model.X[i,j]**2)*model.I[i,j,t] - model.V[j,t] == 0)
model.voltage_drop = Constraint(model.Ol, model.OT, rule=voltage_drop_rule) # Voltage drop

def define_current_rule (model, i, j, t):
    return ((model.I[i,j,t])*(model.V[j,t]) <= model.P[i,j,t]**2 + model.Q[i,j,t]**2)
model.define_current = Constraint(model.Ol, model.OT, rule=define_current_rule) # Power flow

def voltage_limit_rule(model,i,t):
    return (model.Vmin[i] , model.V[i,t], model.Vmax[i])
model.voltage_limit = Constraint(model.Ob, model.OT, rule = voltage_limit_rule)

def current_limit_rule(model,i,j,t):
    return (0,model.I[i,j,t],model.Imax[i,j])
model.current_limit = Constraint(model.Ol, model.OT, rule = current_limit_rule)

# Define Objective Function
def act_loss(model):
    return  (sum(model.CONS_cost[i] for i in model.Ocons)
            +sum(model.DG_cost[i] for i in model.Odg)
            + sum(model.ESS_cost[i] for i in model.Oess)
            + sum(model.PV_cost[i] for i in model.Opv)
            + sum(model.EV_cost[i] for i in model.Oev))
model.obj = Objective(rule=act_loss)

# Run model 
# solver = SolverFactory("gurobi")
# #solver.options['NonConvex'] = 2
# solver.solve(model)



solver = SolverFactory('ipopt')  # couenne

# Solve
result = solver.solve(model, tee=True)
model.display()