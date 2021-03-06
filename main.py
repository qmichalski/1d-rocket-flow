# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 12:38:13 2021

@author: quent
"""

import numpy as np

import geometrylib
import flowsolver
import massflowlib
import matplotlib.pyplot as plt

def initFlow(P0,T0,X0,A,grid,gas,process='quick'):
    from scipy.interpolate import interp1d
    r0 = np.zeros(len(grid))
    e0 = np.zeros(len(grid))
    u0 = np.zeros(len(grid))
    gas.TPX = T0, P0, X0
    # inlet
    rSet = np.zeros(3)
    eSet = np.zeros(3)
    uSet = np.zeros(3)
    if process == 'quick':
        velocity,density,e = gas.sonicFlowArea(P0,T0,X0,A,0)
        rSet[0] = density
        eSet[0] = e
        uSet[0] = velocity
        gas.TPX = T0,P0, X0
        velocity,density,throatPressure = gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
        gas.SP = gas.entropy_mass, throatPressure
        e = gas.int_energy_mass
        rSet[1] = density
        eSet[1] = e
        uSet[1] = velocity
        velocity,density,e = gas.sonicFlowArea(P0,T0,X0,A,-1)
        rSet[2] = density
        eSet[2] = e
        uSet[2] = velocity
        gridSet = [grid[0],grid[np.argmin(A)],grid[-1]]
        f = interp1d(gridSet, rSet, kind='linear')
        r0 = f(grid)
        f = interp1d(gridSet, eSet, kind='linear')
        e0 = f(grid)
        f = interp1d(gridSet, uSet, kind='linear')
        u0 = f(grid)
    if process == 'slow':
        for ii,z in enumerate(A):
            velocity,density,e = gas.sonicFlowArea(P0,T0,X0,A,ii)
            r0[ii] = density
            e0[ii] = e
            u0[ii] = velocity
        
    return(r0,e0,u0)

nbrPointsSet = np.array([21,31,51,71,101,151,201,251,301])
massflowRatioSet = np.zeros(len(nbrPointsSet))
throatMachSet = np.zeros(len(nbrPointsSet))
densityRatioSet = np.zeros(len(nbrPointsSet))
energyRatioSet = np.zeros(len(nbrPointsSet))
velocityRatioSet = np.zeros(len(nbrPointsSet))

for ii,nbrPoint in enumerate(nbrPointsSet):
    print('{} case running'.format(nbrPoint))
    
    geometry = geometrylib.OneDGeometry()
    geometry.andersonConstructor(throatDiameter=0.1,nbrPoints=nbrPoint)
    # geometry.rocketEngineConstructor(combustionChamberLength=200e-3,
    #                                   chamberDiameter=50e-3, 
    #                                   convergentAngle=45, 
    #                                   throatDiameter=25e-3, 
    #                                   roughness=10e-6,
    #                                   nbrPoints=60)
    
    T0 = 2000
    P0 = 25e5
    X0 = 'N2:1'
    mech = 'gri30_highT.cti'
    submech = 'gri30_mix'
    gasTest = massflowlib.Massflow_Solution(mech,submech)
    gasTest.TPX = T0, P0, X0
    velocity,density,throatPressure = gasTest.chokedNozzle(isentropicEfficiency=1, frozen=True)

    A = geometry.crossSection
    throatIndex = np.argmin(A)
    
    massflow0 = velocity*density*A[throatIndex]
    # print(massflow0)
    # fluidModel = 'cantera'
    fluidModel = 'cantera'
    if fluidModel == 'cantera':
        q1DCF = flowsolver.Quasi1DCompressibleFlow(geometry,
                                                   fluidModel=fluidModel,
                                                   mech=mech, 
                                                   submech=submech,
                                                   Tw=300,
                                                   wallHeatFluxCorrelation='adiabatic', # bartz,adiabatic
                                                   wallShearStressCorrelation='non-viscous') # non-viscous, moody_bulk
        q1DCF.gas.TPX = T0, P0, X0
        e00 = q1DCF.gas.int_energy_mass
        r00 = q1DCF.gas.density
        a00 = q1DCF.gas.soundSpeed()
        cp00 = q1DCF.gas.cp_mass
        u00 = massflow0 / (r00*A[0])
        h00 = q1DCF.gas.enthalpy_mass
        ht0 = h00 + 0.5*u00**2
        s00 = q1DCF.gas.entropy_mass
        q1DCF.gas.HS = ht0, s00
    
    r0,e0,u0 = initFlow(P0,T0,X0,geometry.crossSection,geometry.grid,q1DCF.gas,process='quick')
    
    initialTimeState = {'r': r0,'u': u0,'e': e0}
    
    q1DCF.setInitialTimeState(initialTimeState)
    print('{} case initialized'.format(nbrPoint))
    
    sol = q1DCF.solveSteadyQuasi1D(  CFL=0.4,
                                     tol=1e-6, 
                                     maxSteps=None, 
                                     fullOutput=True, 
                                     plot=False,
                                     plotStep=100,
                                     showConvergenceProgress=False,
                                     method='MacCormack' ) #MacCormack
    
    print('{} case solved'.format(nbrPoint))
    
    q1DCF.gas.HS = ht0, s00
    velocity,density,throatPressure = q1DCF.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
    massflow0 = velocity*density*A[throatIndex]
    
    r0,e0,u0 = initFlow(P0,T0,X0,geometry.crossSection,geometry.grid,q1DCF.gas,process='slow')
    
    massflowRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.r[throatIndex]*q1DCF.u[throatIndex]*q1DCF.A[throatIndex]/massflow0-1)
    densityRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.r/r0-1)
    energyRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.e/e0-1)
    velocityRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.u/u0-1)
    
    massflowRatioSet[ii] = massflowRatio
    densityRatioSet[ii] = densityRatio
    energyRatioSet[ii] = energyRatio
    velocityRatioSet[ii] = velocityRatio
    throatMachSet[ii] = q1DCF.M[throatIndex]
    
    print('Area at throat : {}'.format(q1DCF.A[throatIndex]))
    print('Area derivative around throat : ({},{}) '.format(q1DCF.dlnA_dx[throatIndex],q1DCF.dlnA_dx[throatIndex+1]))
    print('Mach number at throat : {}'.format(q1DCF.M[throatIndex]))
    print('Grid is {} points - Massflow residual = {}'.format(q1DCF.nbrPoints,massflowRatio))
    plt.plot(sol['iterations'],sol['max_residuals'])
    plt.xlabel('Iteration [-]')
    plt.ylabel('Max of normalized residuals [-]')
    plt.yscale('log')
    plt.show()

fig,ax = plt.subplots(1,1,figsize=(6, 6), dpi=400)
plt.plot(nbrPointsSet,abs(throatMachSet-1),'-d',label='Residual of Mach at throat')
plt.plot(nbrPointsSet,abs(massflowRatioSet),'-o',label='Residual of mass flow')
plt.plot(nbrPointsSet,abs(densityRatioSet),'-s',label='Residual of density')
plt.plot(nbrPointsSet,abs(energyRatioSet),'-o',label='Residual of internal energy')
plt.plot(nbrPointsSet,abs(velocityRatioSet),'-s',label='Residual of velocity')
plt.legend()
plt.xlabel('Number of grid points [points]')
plt.ylabel('Residual [-]')
plt.xscale('log')
plt.yscale('log')
plt.savefig('./validation/fig/residuals_anderson.png',dpi=400)
plt.show()

    
    
    