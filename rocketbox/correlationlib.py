# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:41:48 2021

@author: quent
"""

def wallHeatFluxCorrelation(gas, parameters):
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
        correlation_list = ['bartz','adiabatic']
        correlation = parameters['wallHeatFluxCorrelation']
        if correlation in correlation_list:
            if correlation == 'bartz':
                h = bartz(gas, parameters)
            if correlation == 'adiabatic':
                h = 0
        else:
            print('Method is not in method list ({})'.format(correlation_list))
        return(h)   

def bartz(gas, parameters):
    Tw = parameters['wallTemperature']
    D = parameters['chamberDiameter']
    U = parameters['bulkFlowVelocity']
    C = 0.026 # coefficient of Nusselt correlation fitted by Bartz
    T = gas.T # bulk static temperature
    T_am = 0.5*(T+Tw) # film temperature: average between bulk and wall
    gas.TP = T_am,gas.P # updating fluid to average temperature
    rho_am = gas.density # film temperature density
    mu_am = gas.viscosity # film temperature viscosity 
    cp_am = gas.cp_mass # film temperature heat capacity
    lambda_am = gas.thermal_conductivity # film temperature thermal conductivity
    Pr_am = cp_am * mu_am / lambda_am # film temperature Prandtl number
    h = C / (D**(0.2)) * (mu_am**(0.2) * cp_am / (Pr_am**(0.6))) * (rho_am * U)**(0.8)
    return(h)

def wallShearStressCorrelation(gas,parameters):
    correlation_list = ['non-viscous','moody_bulk']
    correlation = parameters['wallShearStressCorrelation']
    if correlation in correlation_list:
        if correlation == 'moody_bulk':
            tau_wall = moodyBulk(gas,parameters)
        if correlation == 'non-viscous':
            tau_wall = 0
        return(tau_wall)
    else:
        print('Method is not in method list ({})'.format(correlation_list))

def moodyBulk(gas,parameters):
    from numpy import log10
    # Assume that the lengthscale applicable to the correlation is 
    # the cross-sectional area divided by the hydraulic perimeter
    U = parameters['bulkFlowVelocity']
    A = parameters['area']
    Ph = parameters['hydraulicPerimeter']
    r = parameters['roughness']
    hydraulicRadius = A/Ph
    hydraulicDiameter = 4*hydraulicRadius
    lengthscale = hydraulicDiameter
    eL = r/lengthscale
    Re_L = gas.density*U*lengthscale/gas.viscosity
    f_Darcy = ( -1.8 * log10( 6.9 / Re_L + (eL / 3.7) ** 1.11 ))**(-2)
    f_Fanning = f_Darcy/4
    tau_wall = f_Fanning*gas.density*U**2/2
    return tau_wall

def _test_moodyBulkCorrelation():
    import massflowlib
    import numpy as np
    fluid = massflowlib.Massflow_Solution('gri30_highT.xml','gri30_mix') 
    T0 = 300
    p0 = 200e5
    fluid.TPX = T0, p0, 'CH4:1,O2:2'
    fluid.equilibrate('HP')
    mdot = 1 # kg/s
    D = 50e-3 # m
    A = np.pi*(D/2)**2
    Ph = np.pi*D
    rho = fluid.density
    U = mdot / (A * rho)
    r = 10e-6
    parameters={'wallShearStressCorrelation':'Moody_bulk',
                'area':A,
                'bulkFlowVelocity':U,
                'hydraulicPerimeter':Ph,
                'roughness':r}
    tau_wall = wallShearStressCorrelation(fluid,parameters)
    print('Tau_wall : {} J/m3'.format(tau_wall))
    return()
    
def _test_bartzCorrelation():
    # Testing with burn products of stoichiometric methane-oxygen combustion
    # at 10 bar and 1 kg/s of total mass flow in a 50 mm combustor
    # wall temperature is 500 K
    import massflowlib
    import numpy as np
    fluid = massflowlib.Massflow_Solution('gri30_highT.xml','gri30_mix')
    T0 = 300
    p0 = 200e5
    fluid.TPX = T0, p0, 'CH4:1,O2:2'
    # drh = enthalpyOfReaction(fluid,T0,p0)
    # fluid.TPX = T0, p0, 'CH4:1,O2:2'
    # print(drh)
    fluid.equilibrate('HP')
    rho = fluid.density 
    T = fluid.T
    a = fluid.soundSpeed(frozen=True)
    mdot = 1 # kg/s
    D = 50e-3 # m
    S = np.pi*(D/2)**2
    U = mdot / (S * rho)
    M = U/a
    Tw = 500 # K
    parameters={'wallHeatFluxCorrelation':'Bartz',
                'wallTemperature':Tw,
                'bulkFlowVelocity':U,
                'chamberDiameter':D}
    h = wallHeatFluxCorrelation(fluid, parameters)
    q = h * (T - Tw)
    # alpha = q * A / abs(drh * mdot)
    print('Mach: {} | Velocity: {} m/s | Diameter {} mm | Wall temperature : {} K | Convection coef. : {} W/m2/K | Wall heat flux : {} MW/m2 %'.format(M,U,D*1e3,Tw,h,q/1e6))
    return()