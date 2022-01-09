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
Afun = lambda x: 1 + 2.2*(x-1.5)**2

length = 3
# nbrPointsSet = np.array([21,31,51,71,101,201,301])
nbrPointsSet = np.array([51])
massflowRatioSet = np.zeros(len(nbrPointsSet))
throatMachSet = np.zeros(len(nbrPointsSet))

for ii,nbrPoint in enumerate(nbrPointsSet):
 
    geometry = geometrylib.OneDGeometry()
    geometry.andersonConstructor(nbrPoints=51)
    # geometry.rocketEngineConstructor(combustionChamberLength=200e-3,
                                     # chamberDiameter=50e-3, 
                                     # convergentAngle=45, 
                                     # throatDiameter=25e-3, 
                                     # roughness=10e-6,
                                     # nbrPoints=60)
    
    T0 = 2000
    p0 = 25e5
    X0 = 'N2:1'
    mech = 'gri30_highT.cti'
    submech = 'gri30_mix'
    gasTest = massflowlib.Massflow_Solution(mech,submech)
    gasTest.TPX = T0, p0, X0
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
                                                   wallHeatFluxCorrelation='bartz',
                                                   wallShearStressCorrelation='moody_bulk') # non-viscous
        q1DCF.gas.TPX = T0, p0, X0
        e00 = q1DCF.gas.int_energy_mass
        r00 = q1DCF.gas.density
        a00 = q1DCF.gas.soundSpeed()
        cp00 = q1DCF.gas.cp_mass
        u00 = massflow0 / (r00*A[0])
        h00 = q1DCF.gas.enthalpy_mass
        ht0 = h00 + 0.5*u00**2
        s00 = q1DCF.gas.entropy_mass
        q1DCF.gas.HS = ht0, s00
    
    if fluidModel == 'perfectgas':
        Mgas = 0.028
        k = 1.25
        q1DCF = flowsolver.Quasi1DCompressibleFlow(geometry,
                                                   fluidModel=fluidModel,
                                                   k=k,
                                                   Mgas=Mgas,
                                                   wallHeatFluxCorrelation='adiabatic',
                                                   wallShearStressCorrelation='non-viscous')
        rgas = 8.314/Mgas
        cv = rgas/(k-1)
        r00 = p0/(rgas*T0)
        e00 = cv*T0
    
        # velocity,density,throatPressure = q1DCF.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
        # massflow0 = velocity*density*A[throatIndex]
    # print(massflow0)
    
    rfun = lambda x: r00*(-0.3146*x+1)
    efun = lambda x: e00*(-0.2314*x+1)
    ufun = lambda x: massflow0/(rfun(x)*Afun(x))
    
    r0 = rfun(geometry.grid)
    u0 = ufun(geometry.grid)
    e0 = efun(geometry.grid)
    
    initialTimeState = {'r': r0,'u': u0,'e': e0}
    
    q1DCF.setInitialTimeState(initialTimeState)
    
    sol = q1DCF.solveSteadyQuasi1D(  CFL=0.4,
                                     tol=1e-6, 
                                     maxSteps=None, 
                                     fullOutput=True, 
                                     plot=True,
                                     plotStep=100,
                                     showConvergenceProgress=False,
                                     method='MacCormack' ) #MacCormack
    
    q1DCF.gas.HS = ht0, s00
    velocity,density,throatPressure = q1DCF.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
    massflow0 = velocity*density*A[throatIndex]
    
    massflowRatio = np.linalg.norm(q1DCF.r[throatIndex]*q1DCF.u[throatIndex]*q1DCF.A[throatIndex])/massflow0
    massflowRatioSet[ii] = massflowRatio
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
    
    plt.plot(nbrPointsSet,abs(throatMachSet-1)*100,'-d',label='Residual of Mach at throat')
    plt.plot(nbrPointsSet,abs(massflowRatioSet-1)*100,'-o',label='Residual of mass flow')
    plt.legend()
    plt.show()

