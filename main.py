# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 12:38:13 2021

@author: quent
"""

import numpy as np

import flowsolver
import massflowlib
import matplotlib.pyplot as plt

Afun = lambda x: 1 + 2.2*(x-1.5)**2

length = 3
grid = np.linspace(0,length,61)

T0 = 2000
p0 = 25e5
X0 = 'N2:1'
mech = 'gri30_highT.cti'

gasTest = massflowlib.Massflow_Solution(mech)
gasTest.TPX = T0, p0, X0
velocity,density,throatPressure = gasTest.chokedNozzle(isentropicEfficiency=1, frozen=True)
A = Afun(grid)
throatIndex = np.argmin(A)

massflow0 = velocity*density*A[throatIndex]
# print(massflow0)
# fluidModel = 'cantera'
fluidModel = 'cantera'
if fluidModel == 'cantera':
    q1DCF = flowsolver.Quasi1DCompressibleFlow(grid,A=A,fluidModel=fluidModel,mech=mech)
    
    Tfun = lambda x: T0*(-1/3*x+1)
    p00 = p0
    
    q1DCF.gas.TPX = Tfun(0), p00, X0
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
    q1DCF = flowsolver.Quasi1DCompressibleFlow(grid,A=A,fluidModel=fluidModel)
    rgas = 8.314/0.028
    cv = rgas/(1.4-1)
    r00 = p0/(rgas*T0)
    e00 = cv*T0

    # velocity,density,throatPressure = q1DCF.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
    # massflow0 = velocity*density*A[throatIndex]
# print(massflow0)

rfun = lambda x: r00*(-0.3146*x+1)
efun = lambda x: e00*(-0.2314*x+1)
ufun = lambda x: massflow0/(rfun(x)*Afun(x))

r0 = rfun(grid)
u0 = ufun(grid)
e0 = efun(grid)

initialTimeState = {'r': r0,'u': u0,'e': e0}

q1DCF.setInitialTimeState(initialTimeState)

sol = q1DCF.solveSteadyQuasi1D(  CFL=0.5,
                                 tol=1e-6, 
                                 maxSteps=10000, 
                                 fullOutput=True, 
                                 plot=True,
                                 plotStep=1000,
                                 showConvergenceProgress=False,
                                 method='MacCormack-1'    ) #MacCormack

q1DCF.gas.TP = q1DCF.T[0],q1DCF.p[0]
velocity,density,throatPressure = q1DCF.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
massflow0 = velocity*density*A[throatIndex]

massflowRatio = q1DCF.r[throatIndex]*q1DCF.u[throatIndex]*q1DCF.A[throatIndex]/massflow0

print('Area at throat : {}'.format(q1DCF.A[throatIndex]))
print('Area derivative around throat : ({},{}) '.format(q1DCF.dlnA_dx[throatIndex],q1DCF.dlnA_dx[throatIndex+1]))
print('Mach number at throat : {}'.format(q1DCF.M[throatIndex]))

plt.plot(sol['iterations'],sol['max_residuals'])
plt.xlabel('Iteration [-]')
plt.ylabel('Max of normalized residuals [-]')
plt.yscale('log')

