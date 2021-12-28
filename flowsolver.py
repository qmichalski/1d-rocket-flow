# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:27:42 2021

@author: quent
"""

import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import massflowlib

__version__ = 0.1

class Quasi1DCompressibleFlow():
    def __init__(
        self, grid, A, fluidModel='cantera', mech='gri30_highT.cti'
    ):
        self.fluidModel_list = ['cantera','perfectgas']
        if fluidModel in self.fluidModel_list:
            self.fluidModel = fluidModel
            if fluidModel == 'cantera':
                self.gas = massflowlib.Massflow_Solution(mech)
        else:
            print('Model not in model list')
            print('Models available are {}'.format(self.fluidModel))
        self.grid = grid # spatial discretisation of the 1d problem
        self.nbrPoints = len(grid) # number of points in grid
        self.A = A  # area
        self.p = np.zeros(self.nbrPoints) # static pressure array
        self.r = np.zeros(self.nbrPoints) # static density array
        self.e = np.zeros(self.nbrPoints) # static internal energy array
        self.u = np.zeros(self.nbrPoints) # velocity array
        self.h = np.zeros(self.nbrPoints) # static enthalpy array
        self.T = np.zeros(self.nbrPoints) # static temperature array
        self.cv = np.zeros(self.nbrPoints) # static temperature array
        self.M = np.zeros(self.nbrPoints) # Mach number array
        self.a = np.zeros(self.nbrPoints) # local sound speed
        self.SV_names = ['r','u','e'] # state variable list
        self.SV = np.array([self.r,self.u,self.e]) # stores the state variables
        self.SV_nbr = len(self.SV_names) # number of state variables
        self.dx = self.grid[1]-self.grid[0] # spatial step
        # Store the residual value of the state variables
        self.SV_residuals = np.array([np.ones(self.nbrPoints),
                                      np.ones(self.nbrPoints), 
                                      np.ones(self.nbrPoints)])
        
    def setInitialTimeState(self,initialState):
        for stateVariable in initialState:
            if len(initialState[stateVariable]) == self.nbrPoints:
                if stateVariable == 'r':
                    self.r = initialState[stateVariable]
                if stateVariable == 'u':
                    self.u = initialState[stateVariable]
                if stateVariable == 'e':
                    self.e = initialState[stateVariable]
                self.SV = np.array([self.r,self.u,self.e])
            else:
                print('Wrong number of points in {} during initialization'.format(stateVariable))
                
    def _dSV_dx(self,method):
        if method == 'backward-1':
            SV_spatialBackwardDiff = np.zeros((3,self.nbrPoints))
            SV_spatialBackwardDiff[:,1:] = (self.SV[:,1:] - self.SV[:,0:-1])/self.dx
            self.dSV_dx = SV_spatialBackwardDiff
            dlnA_dx = np.zeros(self.nbrPoints)
            dlnA_dx[1:] = (np.log(self.A[1:]) - np.log(self.A[0:-1]))/self.dx
            self.dlnA_dx = dlnA_dx
        if method == 'forward-1':
            SV_spatialForwardDiff = np.zeros((3,self.nbrPoints))
            SV_spatialForwardDiff[:,:-1] = (self.SV[:,1:] - self.SV[:,0:-1])/self.dx
            self.dSV_dx = SV_spatialForwardDiff
            dlnA_dx = np.zeros(self.nbrPoints)
            dlnA_dx[:-1] = (np.log(self.A[1:]) - np.log(self.A[0:-1]))/self.dx
            self.dlnA_dx = dlnA_dx

    def _SV(self,SV):
        '''
        Update all gas values based on state variables
        '''
        self.SV = SV
        for ii,z in enumerate(np.transpose(SV)):
            self.r[ii] = z[0]
            self.u[ii] = z[1]
            self.e[ii] = z[2]
            self.gas.UV = z[2], 1/z[0]
            self.a[ii] = self.gas.soundSpeed(frozen=True)
            self.p[ii] = self.gas.P
            self.T[ii] = self.gas.T
            self.h[ii] = self.gas.enthalpy_mass
            self.M[ii] = z[1]/self.a[ii]
            self.cv[ii] = self.gas.cv_mass
     
    def _dSV_dt(self,method):
        # Making sure all variables are updated
        dSV_dt = np.zeros((3,self.nbrPoints))
        dx = self.dx
        A = self.A
        self._dSV_dx(method)
        dlnA_dx = self.dlnA_dx
        dSV_dx = self.dSV_dx
        for ii,z in enumerate(zip(self.r,self.u,self.e,self.p,self.T,self.cv)):
            r = z[0]
            u = z[1]
            e = z[2]
            p = z[3]
            T = z[4]
            cv = z[5]
            M = [[u         ,r    ,0        ],
                 [p/(r**2)  ,u    ,p/(r*e) ],
                 [0         ,p/r  ,u        ]]
            C = [r*u*dlnA_dx[ii],
                 0,
                 p*u/r*dlnA_dx[ii]]
            dSV_dt[:,ii] = - np.matmul(M, np.transpose(dSV_dx[:,ii])) - C
        self.dSV_dt = dSV_dt

    def _integrationStep(self,CFL,method='MacCormack-1'):
        # Adaptive dt for stability using constant CFL
        dt = CFL*np.min(self.dx/(self.u+self.a))
        initial_SV = self.SV
        if method == 'MacCormack-1':
            # Calculation of the predictor
            self._dSV_dt('forward-1')
            predictor_dSV_dt = self.dSV_dt
            predictor_SV = initial_SV + predictor_dSV_dt*dt
            # evaluation of SV at predictor values
            self._SV(predictor_SV)
            # Calculation of the corrector from predictor values
            self._dSV_dt('backward-1')
            corrector_dSV_dt = self.dSV_dt
            # Calculation of the final time derivative (second order accuracy)
            averaged_dSV_dt = 0.5*(predictor_dSV_dt+corrector_dSV_dt)
            final_SV = initial_SV + averaged_dSV_dt*dt
            
            # Applying boundary conditions
            final_SV[0,0]=initial_SV[0,0] # set density
            final_SV[1,0]=2*final_SV[1,1]-final_SV[1,2] # zero grad conditions
            final_SV[2,0]=initial_SV[2,0] # set internal energy
            # Zero grad conditions at outlet
            final_SV[0,-1]=2*final_SV[0,-2]-final_SV[0,-3] 
            final_SV[1,-1]=2*final_SV[1,-2]-final_SV[1,-3]
            final_SV[2,-1]=2*final_SV[2,-2]-final_SV[2,-3]
            
        if method == 'RK45':
            # Calculation of the predictor
            self._dSV_dt('forward-1') 
            k1 = self.dSV_dt
            ynk1 = initial_SV + k1*dt/2
            self._SV(ynk1)
            self._dSV_dt('backward-1') 
            k2 = self.dSV_dt
            ynk2 = initial_SV + k2*dt/2
            self._SV(ynk2)
            self._dSV_dt('backward-1') 
            k3 = self.dSV_dt
            ynk3 = initial_SV + k3*dt
            self._SV(ynk3)
            self._dSV_dt('backward-1') 
            k4 = self.dSV_dt
            averaged_dSV_dt = 1/6*(k1 + 2*k2 + 2*k3 + k4)*dt
            final_SV = initial_SV + averaged_dSV_dt*dt
            
            # Applying boundary conditions
            final_SV[0,0]=initial_SV[0,0] # set density
            final_SV[1,0]=2*final_SV[1,1]-final_SV[1,2] # zero grad conditions
            final_SV[2,0]=initial_SV[2,0] # set internal energy
            # Zero grad conditions at outlet
            final_SV[0,-1]=2*final_SV[0,-2]-final_SV[0,-3] 
            final_SV[1,-1]=2*final_SV[1,-2]-final_SV[1,-3]
            final_SV[2,-1]=2*final_SV[2,-2]-final_SV[2,-3]
            
        self._SV(final_SV)
        # Calculation of the residuals (dSV_dt*dt/SV)
        SV_residuals = abs(final_SV-initial_SV) / initial_SV
        SV_residuals[0,0] = np.nan
        SV_residuals[2,0] = np.nan
        self.SV_residuals = SV_residuals
            
    def solveSteadyQuasi1D(self, 
                           CFL=0.3, 
                           tol=1e-6, 
                           maxSteps=None, 
                           fullOutput=False, 
                           plot=False,
                           plotStep=100,
                           showConvergenceProgress=False,
                           method='MacCormack'):
        '''
        Performs the integration of 
        '''
        residuals = []
        step = 0
        maximumResidual = np.max(self.SV_residuals[:,1:].flatten())
        while maximumResidual>tol:
            self._integrationStep(CFL, method=method)
            maximumResidual = np.max(self.SV_residuals[:,1:].flatten())
            residuals.append(maximumResidual)
            step += 1
            if showConvergenceProgress:
                print(maximumResidual,
                      np.unravel_index(np.argmax(self.SV_residuals[:,1:]),
                                       self.SV_residuals[:,1:].shape))
            if step == maxSteps:
                break
            if plot and np.mod(step,plotStep)==0:
                if 'fig' in locals():
                    plt.rcParams['xtick.labelbottom'] = False
                    ii=0
                    ax[ii].plot(self.grid,self.r,label=str(step))
                    ax[ii].set_ylabel('Density [kg/m3]')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.u,label=str(step))
                    ax[ii].set_ylabel('Velocity [m/s]')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.e,label=str(step))
                    ax[ii].set_ylabel('Internal energy [J/kg]')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.p/1e5,label=str(step))
                    ax[ii].set_ylabel('Pressure [bar]')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.T,label=str(step))
                    ax[ii].set_ylabel('Temperature [K]')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.r*self.u*self.A,label=str(step))
                    ax[ii].set_ylabel('Mass flow [kg/s]')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.M,label=str(step))
                    ax[ii].set_ylabel('Mach [-]')
                    ax[ii].tick_params(axis='x',labelbottom='off') # labels along the bottom edge are off
                    ii=ii+1
                    ax[ii].plot(self.grid,self.SV_residuals[0,:],label=str(step))
                    ax[ii].set_ylabel('Mass equation residuals [-]')
                    ax[ii].set_yscale('log')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.SV_residuals[1,:],label=str(step))
                    ax[ii].set_ylabel('Momentum equation residuals [-]')
                    ax[ii].set_yscale('log')
                    ii=ii+1
                    ax[ii].plot(self.grid,self.SV_residuals[2,:],label=str(step))
                    ax[ii].set_ylabel('Energy equation residuals [-]')
                    ax[ii].set_yscale('log')
                    ax[ii].tick_params(axis='x',labelbottom='off') # labels along the bottom edge are off
                    ax[ii].set_xlabel('Axial position [m]')
                else:
                    fig,ax = plt.subplots(10,1,figsize=(6, 10), dpi=400)
        plt.show()
        plt.rcParams['xtick.labelbottom'] = True
        if fullOutput:
            sol = {}
            sol['max_residuals'] = np.array(residuals)
            sol['iterations'] = np.linspace(0,step-1,step)
            return(sol)

def _test__dSV_dx():
    # Test of the differentiation method
    # Check that the array method returns the same output as the 
    # loop method where the scheme is more explicit
    Tfun = lambda x: 2000*(1 - 1/3*x)**2    
    length = 2.9
    grid = np.linspace(0,length,30)
    dx = grid[1] - grid[0]
    SV_array = Tfun(grid)
    # plt.plot(grid,SV_array)
    SV_spatialBackwardDiff_arrayCalc = np.zeros(len(grid))
    SV_spatialForwardDiff_arrayCalc = np.zeros(len(grid))
    SV_spatialBackwardDiff_loop = np.zeros(len(grid))
    SV_spatialForwardDiff_loop = np.zeros(len(grid))
    SV_spatialBackwardDiff_arrayCalc[1:] = SV_array[1:] - SV_array[0:-1]
    SV_spatialForwardDiff_arrayCalc[:-1] = SV_array[1:] - SV_array[0:-1]
    for ii,SV in enumerate(SV_array):
        if ii > 0: 
            SV_spatialBackwardDiff_loop[ii] = SV_array[ii] - SV_array[ii-1]
        if ii < (len(SV_array)-1):
            SV_spatialForwardDiff_loop[ii] = SV_array[ii+1] - SV_array[ii]
    SV_spatialBackwardDiff_arrayCalc[0] = 2*SV_spatialBackwardDiff_arrayCalc[1]-SV_spatialBackwardDiff_arrayCalc[2] # zeros gradient condition
    SV_spatialBackwardDiff_loop[0] = 2*SV_spatialBackwardDiff_loop[1]-SV_spatialBackwardDiff_loop[2]
    SV_spatialForwardDiff_arrayCalc[-1] = 2*SV_spatialForwardDiff_arrayCalc[-2]-SV_spatialForwardDiff_arrayCalc[-3] # zeros gradient condition
    SV_spatialForwardDiff_loop[-1] = 2*SV_spatialForwardDiff_loop[-2]-SV_spatialForwardDiff_loop[-3]
    plt.plot(grid,SV_spatialBackwardDiff_arrayCalc,label='SV_spatialBackwardDiff_arrayCalc')
    plt.plot(grid,SV_spatialForwardDiff_arrayCalc,label='SV_spatialForwardDiff_arrayCalc')
    plt.plot(grid,SV_spatialBackwardDiff_loop,'o',label='SV_spatialBackwardDiff_loop')
    plt.plot(grid,SV_spatialForwardDiff_loop,'o',label='SV_spatialForwardDiff_loop')
    plt.legend()
    return()