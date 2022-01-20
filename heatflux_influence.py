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

# nbrPointsSet = np.array([5,10,20,30,50,100])/1e3
nbrPointsSet = np.array([0.005])

massflowRatioSet = np.zeros(len(nbrPointsSet))
throatMachSet = np.zeros(len(nbrPointsSet))
densityRatioSet = np.zeros(len(nbrPointsSet))
energyRatioSet = np.zeros(len(nbrPointsSet))
velocityRatioSet = np.zeros(len(nbrPointsSet))
timeSet = np.zeros(len(nbrPointsSet))

for ii,nbrPoint in enumerate(nbrPointsSet):
    D0 = nbrPoint
    print('{} case running'.format(nbrPoint))
    
    geometry = geometrylib.OneDGeometry()
    # geometry.andersonConstructor(throatDiameter=D0,nbrPoints=51)
    geometry.rocketEngineConstructor(combustionChamberLength=10e-3,
                                      chamberDiameter=25e-3,
                                      convergentAngle=45,
                                      throatDiameter=15e-3,
                                      nozzleDiameter=20e-3,
                                      nozzleLength=30e-3,
                                      roughness=10e-6,
                                      nbrPoints=101,
                                      nozzleType='bellApproximation',plot=True)
    
    T0 = 300
    P0 = 5e5
    X0 = 'CH4:1,O2:2'
    # X0 = 'N2:1'
    mech = 'gri30_highT.xml'
    submech = 'gri30_mix'
    gasTest = massflowlib.Massflow_Solution(mech,submech)
    gasTest.TPX = T0, P0, X0
    DrH = gasTest.heatOfReaction(reactionType='HP')
    gasTest.equilibrate('HP')
    T0 = gasTest.T
    X0 = gasTest.X
    velocity,density,throatPressure = gasTest.chokedNozzle(isentropicEfficiency=1, frozen=True)

    A = geometry.crossSection
    throatIndex = np.argmin(A)
    
    massflow0 = velocity*density*A[throatIndex]
    
    q1DCF_nonideal = flowsolver.Quasi1DCompressibleFlow(geometry,
                                                        fluidModel='cantera',
                                                        mech=mech, 
                                                        submech=submech,
                                                        Tw=600,
                                                        wallHeatFluxCorrelation='bartz', # bartz,adiabatic
                                                        wallShearStressCorrelation='non-viscous') # non-viscous, moody_bulk
     
    q1DCF_nonideal.initSonicRocketFlow(P0,T0,X0,geometry.crossSection,geometry.grid,process='quick')
    u0=q1DCF_nonideal.u
    r0=q1DCF_nonideal.r
    print('{} case initialized'.format(nbrPoint))
    
    t1 = time()
    sol = q1DCF_nonideal.solveSteadyQuasi1D(  CFL=0.5,
                                              tol=1e-6,
                                              maxSteps=None,
                                              fullOutput=True,
                                              plot=False,
                                              plotStep=100,
                                              showConvergenceProgress=False,
                                              method='MacCormack' ) #MacCormack
    
    t2 = time()
    print( '{} case solved'.format(nbrPoint) )
    
    ht = q1DCF_nonideal.h+0.5*q1DCF_nonideal.u**2
    massflow = q1DCF_nonideal.r*q1DCF_nonideal.u*q1DCF_nonideal.Geometry.crossSection
    print('D0 {} mm | P0 {} bar | Energy loss {}%'.format(D0*1e3,P0,(ht[-1]-ht[0])/DrH*100))
    # # q1DCF.gas.HS = ht0, s00
    # # velocity,density,throatPressure = q1DCF.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
    # # massflow0 = velocity*density*A[throatIndex]

    # # # massflowRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.r[throatIndex]*q1DCF.u[throatIndex]*q1DCF.A[throatIndex]/massflow0-1)
    # # # densityRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.r/r0-1)
    # # # energyRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.e/e0-1)
    # # # velocityRatio = 1/q1DCF.nbrPoints*np.linalg.norm(q1DCF.u/u0-1)
    
    # # massflowRatioSet[ii] = massflowRatio
    # # densityRatioSet[ii] = densityRatio
    # # energyRatioSet[ii] = energyRatio
    # # velocityRatioSet[ii] = velocityRatio
    # # throatMachSet[ii] = q1DCF.M[throatIndex]
    # # timeSet[ii] = t2 - t1
    
    # # print('Area at throat : {}'.format(q1DCF.A[throatIndex]))
    # # print('Area derivative around throat : ({},{}) '.format(q1DCF.dlnA_dx[throatIndex],q1DCF.dlnA_dx[throatIndex+1]))
    # # print('Mach number at throat : {}'.format(q1DCF.M[throatIndex]))
    # # print('Grid is {} points - Massflow residual = {}'.format(q1DCF.nbrPoints,massflowRatio))
    # plt.plot(sol['iterations'],sol['max_residuals'])
    # plt.xlabel('Iteration [-]')
    # plt.ylabel('Max of normalized residuals [-]')
    # plt.yscale('log')
    # plt.show()

    # ht0 = h0 + 0.5*(u0**2)
    # ht = q1DCF.h+0.5*q1DCF.u**2
    # plt.plot(q1DCF.Geometry.grid,(ht0-ht)/ht0*100)
    
# plt.plot(q1DCF.Geometry.grid,e0)
# plt.rcParams["font.family"] = 'Times New Roman'
# plt.rcParams["font.size"] = 12
# fig,ax = plt.subplots(1,1,figsize=(6, 6), dpi=400)
# RC = np.max(q1DCF.Geometry.hydraulicDiameter)
# length = q1DCF.Geometry.grid[-1]
# plt.figure(figsize=(6, 6*5*RC/(length*1.2)), dpi=400)
# plt.plot(q1DCF.Geometry.grid,q1DCF.Geometry.hydraulicDiameter,color='k')
# plt.plot(q1DCF.Geometry.grid,-q1DCF.Geometry.hydraulicDiameter,color='k')
# plt.xlim([0,length*1.2])
# plt.ylim([-RC*2.5,RC*2.5])
# plt.xlabel('Axial position [m]')
# plt.ylabel('Geometry radius [m]')
# plt.savefig('./rocketbox/validation/fig/anderson_geometry.png',dpi=400)
# plt.show()

# fig,ax = plt.subplots(1,1,figsize=(6, 6), dpi=400)
# plt.plot(nbrPointsSet,abs(throatMachSet-1),'-d',label='Residual of Mach at throat')
# plt.plot(nbrPointsSet,abs(massflowRatioSet),'-o',label='Residual of mass flow')
# plt.plot(nbrPointsSet,abs(densityRatioSet),'-s',label='Residual of density')
# plt.plot(nbrPointsSet,abs(energyRatioSet),'-o',label='Residual of internal energy')
# plt.plot(nbrPointsSet,abs(velocityRatioSet),'-s',label='Residual of velocity')
# plt.legend()
# plt.xlabel('Number of grid points [points]')
# plt.ylabel('Residual [-]')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('./rocketbox/validation/fig/anderson_residuals.png',dpi=400)
# plt.show()
    
# fig,ax = plt.subplots(1,1,figsize=(6, 6), dpi=400)
# plt.plot(nbrPointsSet,timeSet,'o-',color='black')
# plt.xlabel('Number of grid points [points]')
# plt.ylabel('Time of execution [s]')
# plt.savefig('./rocketbox/validation/fig/anderson_runtime.png',dpi=400)
# plt.show()