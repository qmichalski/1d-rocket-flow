# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 12:38:13 2021

@author: quent
"""

from rocketbox import geometrylib
from rocketbox import flowsolver
from rocketbox import massflowlib

import numpy as np
import matplotlib.pyplot as plt
from time import time

nbrPointsSet = np.array([21,31,51,71,101,151,201,251])
# nbrPointsSet = np.array([151])

massflowRatioSet = np.zeros(len(nbrPointsSet))
throatMachSet = np.zeros(len(nbrPointsSet))
densityRatioSet = np.zeros(len(nbrPointsSet))
energyRatioSet = np.zeros(len(nbrPointsSet))
velocityRatioSet = np.zeros(len(nbrPointsSet))
timeSet = np.zeros(len(nbrPointsSet))

for ii,nbrPoint in enumerate(nbrPointsSet):
    print('{} case running'.format(nbrPoint))
    
    geometry = geometrylib.OneDGeometry()
    geometry.andersonConstructor(throatDiameter=0.1,nbrPoints=nbrPoint)
    
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
    
    q1DCF.initSonicRocketFlow(P0,T0,X0,geometry.crossSection,geometry.grid,process='slow')
    r0 = q1DCF.r
    u0 = q1DCF.u
    e0 = q1DCF.e
    
    t1 = time()
    q1DCF.initSonicRocketFlow(P0,T0,X0,geometry.crossSection,geometry.grid,process='quick')
    print('{} case initialized'.format(nbrPoint))
    
    sol = q1DCF.solveSteadyQuasi1D(  CFL=0.9,
                                     tol=1e-6, 
                                     maxSteps=None, 
                                     fullOutput=True, 
                                     plot=False,
                                     plotStep=100,
                                     showConvergenceProgress=False,
                                     method='MacCormack' ) #MacCormack
    
    t2 = time()
    
    print('{} case solved'.format(nbrPoint))
    
    q1DCF.gas.HS = ht0, s00
    velocity,density,throatPressure = q1DCF.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
    massflow0 = velocity*density*A[throatIndex]

    massflowRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.r[throatIndex]*q1DCF.u[throatIndex]*q1DCF.A[throatIndex]/massflow0-1)
    densityRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.r/r0-1)
    energyRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.e/e0-1)
    velocityRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.u/u0-1)
    
    massflowRatioSet[ii] = massflowRatio
    densityRatioSet[ii] = densityRatio
    energyRatioSet[ii] = energyRatio
    velocityRatioSet[ii] = velocityRatio
    throatMachSet[ii] = q1DCF.M[throatIndex]
    timeSet[ii] = t2 - t1
    
    print('Area at throat : {}'.format(q1DCF.A[throatIndex]))
    print('Area derivative around throat : ({},{}) '.format(q1DCF.dlnA_dx[throatIndex],q1DCF.dlnA_dx[throatIndex+1]))
    print('Mach number at throat : {}'.format(q1DCF.M[throatIndex]))
    print('Grid is {} points - Massflow residual = {}'.format(q1DCF.nbrPoints,massflowRatio))
    plt.plot(sol['iterations'],sol['max_residuals'])
    plt.xlabel('Iteration [-]')
    plt.ylabel('Max of normalized residuals [-]')
    plt.yscale('log')
    plt.show()

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 12
fig,ax = plt.subplots(1,1,figsize=(6, 6), dpi=400)
RC = np.max(q1DCF.Geometry.hydraulicDiameter)
length = q1DCF.Geometry.grid[-1]
plt.figure(figsize=(6, 6*5*RC/(length*1.2)), dpi=400)
plt.plot(q1DCF.Geometry.grid,q1DCF.Geometry.hydraulicDiameter,color='k')
plt.plot(q1DCF.Geometry.grid,-q1DCF.Geometry.hydraulicDiameter,color='k')
plt.xlim([0,length*1.2])
plt.ylim([-RC*2.5,RC*2.5])
plt.xlabel('Axial position [m]')
plt.ylabel('Geometry radius [m]')
plt.savefig('./rocketbox/validation/fig/anderson_geometry.png',dpi=400)
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
plt.savefig('./rocketbox/validation/fig/anderson_residuals.png',dpi=400)
plt.show()
    
fig,ax = plt.subplots(1,1,figsize=(6, 6), dpi=400)
plt.plot(nbrPointsSet,timeSet,'o-',color='black')
plt.xlabel('Number of grid points [points]')
plt.ylabel('Time of execution [s]')
plt.savefig('./rocketbox/validation/fig/anderson_runtime.png',dpi=400)
plt.show()