# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:41:48 2021

@author: quent
"""

def convectionCorrelation(gas, U, D, Tw, method='Bartz'):
        '''
        Compute the convection coefficient based on different correlations:
            
        Bartz correlation:
            
        Bartz, Turbulent bounday-layer heat transfer from rapidly accelerating 
        flow of rocket combustion gases and of heated air, Report 1963
        
        Input list
         gas:   a cantera fluid object that must contain transport properties
         U:     bulk flow velocity in meters/seconds [m/s]
         D:     hydraulic diameter in meters [m]
         Tw:    wall temperature in kelvins [K]
        Output list
         h:     wall convection coefficient with reference to free stream 
                temperatureand wall temperature, in W/m2/K
        '''
        method_list = ['Bartz']
        if method in method_list:
            if method == 'Bartz'
                C = 0.026 # coefficient of Nusselt correlation fitted by Bartz
                T = gas.T # bulk static temperature
                T_am = np.mean([T,Tw]) # film temperature: average between bulk and wall
                gas.TP = T_am,gas.P # updating fluid to average temperature
                rho_am = gas.density # film temperature density
                mu_am = gas.viscosity # film temperature viscosity 
                cp_am = gas.cp_mass # film temperature heat capacity
                lambda_am = gas.thermal_conductivity # film temperature thermal conductivity
                Pr_am = cp_am * mu_am / lambda_am # film temperature Prandtl number
                h = C / (D**(0.2)) * (mu_am**(0.2) * cp_am / (Pr_am**(0.6))) * (rho_am * U)**(0.8)
        else:
            print('Method is not in method list ({})'.format(method_list))
        return(h)   
    
def _test_bartzCorrelation():
    # Testing with burn products of stoichiometric methane-oxygen combustion
    # at 10 bar and 1 kg/s of total mass flow in a 50 mm combustor
    # wall temperature is 500 K
    import cantera as ct
    import massflowlib
    fluid = ct.Solution('gri30_highT.xml','gri30_mix')
    T0 = 300
    p0 = 200e5
    fluid.TPX = T0, p0, 'CH4:1,O2:2'
    # drh = enthalpyOfReaction(fluid,T0,p0)
    fluid.TPX = T0, p0, 'CH4:1,O2:2'
    print(drh)
    fluid.equilibrate('HP')
    rho = fluid.density 
    T = fluid.T
    a = massflowlib.frozenSoundSpeed(fluid)
    mdot = 1 # kg/s
    D = 50e-3 # m
    S = np.pi*(D/2)**2
    U = mdot / (S * rho)
    M = U/a
    Tw = 500 # K
    h = bartzCorrelation(fluid, U, D, Tw)
    q = h * (T - Tw)
    # alpha = q * A / abs(drh * mdot)
    print('Mach: {} | Velocity: {} m/s | Diameter {} mm | Combustor lenght : {} mm | Wall temperature : {} K | Convection coef. : {} W/m2/K | Wall heat flux : {} MW/m2 | Fraction of heat lost: {} %'.format(M,U,D*1e3,Tw,h,q/1e6))
    return()