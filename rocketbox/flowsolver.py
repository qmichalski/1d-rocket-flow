# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:27:42 2021

@author: quent
"""

import numpy as np
import matplotlib.pyplot as plt
from rocketbox import massflowlib
from rocketbox import correlationlib
# import massflowlib
# import correlationlib

__version__ = 0.3 

class Quasi1DCompressibleFlow():
    def __init__(
        self, Geometry, 
        fluidModel='cantera',
        wallHeatFluxCorrelation='adiabatic',
        wallShearStressCorrelation='non-viscous',
        Tw=300,
        mech=None,
        submech=None,
        k=None, 
        Mgas=None
    ):
        self.fluidModel_list = ['cantera','perfectgas']
        if fluidModel in self.fluidModel_list:
            self.fluidModel = fluidModel
            if fluidModel == 'cantera':
                self.gas = massflowlib.Massflow_Solution(mech,submech)
            if fluidModel == 'perfectgas':
                self.k = k
                self.Mgas = Mgas
        else:
            print('Model not in model list')
            print('Models available are {}'.format(self.fluidModel))
        self.Geometry = Geometry
        self.A = self.Geometry.crossSection
        self.nbrPoints = len(Geometry.grid) # number of points in grid
        self.p = np.zeros(self.nbrPoints) # static pressure array
        self.r = np.zeros(self.nbrPoints) # static density array
        self.e = np.zeros(self.nbrPoints) # static internal energy array
        self.u = np.zeros(self.nbrPoints) # velocity array
        self.h = np.zeros(self.nbrPoints) # static enthalpy array
        self.T = np.zeros(self.nbrPoints) # static temperature array
        self.cv = np.zeros(self.nbrPoints) # static temperature array
        self.M = np.zeros(self.nbrPoints) # Mach number array
        self.a = np.zeros(self.nbrPoints) # local sound speed
        self.dpde_rc = np.zeros(self.nbrPoints) # derivative of p relative to e at constant r 
        self.rgas = np.zeros(self.nbrPoints) # gas constant per unit of mass R/m
        # Heat exchange properties
        self.wallHeatFluxCorrelation = wallHeatFluxCorrelation
        self.wallShearStressCorrelation = wallShearStressCorrelation
        self.Tw = Tw*np.ones(self.nbrPoints)
        self.hConv = np.zeros(self.nbrPoints)
        self.dQdx = np.zeros(self.nbrPoints)
        self.flux = np.zeros(self.nbrPoints)
        self.tau_w = np.zeros(self.nbrPoints)
        self.SV_names = ['r','u','e'] # state variable list
        self.SV = np.array([self.r,self.u,self.e]) # stores the state variables
        self.SV_nbr = len(self.SV_names) # number of state variables
        # self.dx = self.Geometry.grid[1]-self.Geometry.grid[0] # spatial step
        # Store the residual value of the state variables
        self.SV_residuals = np.array([np.ones(self.nbrPoints),
                                      np.ones(self.nbrPoints), 
                                      np.ones(self.nbrPoints)])
        self.status = 'created'
        
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
                self._SV(self.SV)
                self.status = 'initialized'
            else:
                print('Wrong number of points in {} during initialization'.format(stateVariable))
                
    def _dSV_dx(self,method):
        '''
        Method
        Spatial differentiation of state variables AND 1D cross-section
        '''
        if method == 'backward':
            SV_spatialBackwardDiff = np.zeros((3,self.nbrPoints))
            SV_spatialBackwardDiff[:,1:] = (self.SV[:,1:] - self.SV[:,0:-1])/(self.Geometry.grid[1:]-self.Geometry.grid[0:-1])
            self.dSV_dx = SV_spatialBackwardDiff
            dlnA_dx = np.zeros(self.nbrPoints)
            dlnA_dx[1:] = (np.log(self.A[1:]) - np.log(self.A[0:-1]))/(self.Geometry.grid[1:]-self.Geometry.grid[0:-1])
            self.dlnA_dx = dlnA_dx
        if method == 'forward':
            SV_spatialForwardDiff = np.zeros((3,self.nbrPoints))
            SV_spatialForwardDiff[:,:-1] = (self.SV[:,1:] - self.SV[:,0:-1])/(self.Geometry.grid[1:]-self.Geometry.grid[0:-1])
            self.dSV_dx = SV_spatialForwardDiff
            dlnA_dx = np.zeros(self.nbrPoints)
            dlnA_dx[:-1] = (np.log(self.A[1:]) - np.log(self.A[0:-1]))/(self.Geometry.grid[1:]-self.Geometry.grid[0:-1])
            self.dlnA_dx = dlnA_dx

    def _SV(self,SV):
        '''
        Method
        Update all gas values based on state variables
        '''
        self.SV = SV
        for ii,z in enumerate(np.transpose(SV)):
            self.r[ii] = z[0]
            self.u[ii] = z[1]
            self.e[ii] = z[2]
            if self.fluidModel == 'cantera':
                self.gas.UV = z[2], 1/z[0]
                self.a[ii] = self.gas.soundSpeed(frozen=True)
                self.p[ii] = self.gas.P
                self.T[ii] = self.gas.T
                self.h[ii] = self.gas.enthalpy_mass
                self.M[ii] = z[1]/self.a[ii]
                self.cv[ii] = self.gas.cv_mass
                self.rgas[ii] = self.gas.gas_constant/self.gas.mean_molecular_weight
                self.dpde_rc[ii] = self.gas.dpde_cr()
            if self.fluidModel == 'perfectgas':
                self.rgas[ii] = 8.314/self.Mgas
                self.cv[ii] = self.rgas[ii]/(self.k-1)
                self.T[ii] = self.e[ii]/self.cv[ii]
                self.a[ii] = ((self.rgas[ii]/self.cv[ii]+1)*self.rgas[ii]*self.T[ii])**(0.5)
                self.p[ii] = self.rgas[ii]*self.r[ii]*self.T[ii]
                self.M[ii] = z[1]/self.a[ii]
                self.dpde_rc[ii] = self.p[ii]/self.e[ii]
    
    def _hConv(self):
        '''
        Method
        Update the value of the convection coefficient at current SV values
        '''
        Tw = self.Tw
        Dh = self.Geometry.hydraulicDiameter
        for ii,z in enumerate(zip(self.r,self.u,self.e)):    
            r = z[0]
            u = z[1]
            e = z[2]
            # updating the gas but might not have to if called before _SV
            # would definitly speed the process not too
            self.gas.UV = e, 1/r
            parameters = {}
            parameters['wallHeatFluxCorrelation'] = self.wallHeatFluxCorrelation
            parameters['wallTemperature'] = Tw[ii]
            parameters['chamberDiameter'] = Dh[ii]
            parameters['bulkFlowVelocity'] = u
            self.hConv[ii] = correlationlib.wallHeatFluxCorrelation(self.gas, parameters)
    
    def _dQdx(self):
        '''
        Method
        Update the value of the differential surface flux at current SV values
        '''
        self._hConv()
        flux = self.hConv*(self.T - self.Tw)
        dQdx = self.Geometry.heatExchangePerimeter*flux
        self.flux = flux
        self.dQdx = dQdx
    
    def _tau_w(self):
        '''
        Method
        Update the value of the wall friction coefficient at current SV values
        '''
        Ph = self.Geometry.hydraulicPerimeter
        roughness = self.Geometry.roughness
        for ii,z in enumerate(zip(self.r,self.u,self.e)):    
            r = z[0]
            u = z[1]
            e = z[2]
            # updating the gas but might not have to if called before _SV
            # would definitly speed the process not too
            self.gas.UV = e, 1/r
            parameters = {}
            parameters['wallShearStressCorrelation'] = self.wallShearStressCorrelation
            parameters['bulkFlowVelocity'] = u
            parameters['area'] = self.A[ii]
            parameters['hydraulicPerimeter'] = Ph[ii]
            parameters['roughness'] = roughness[ii]
            self.tau_w[ii] = correlationlib.wallShearStressCorrelation(self.gas, parameters)
    
    def _dSV_dt(self,method):
        '''
        Method
        Compute the time derivative value
        '''
        dSV_dt = np.zeros((3,self.nbrPoints))
        
        self._dSV_dx(method)
        self._dQdx()
        self._tau_w()
    
        dSV_dx = self.dSV_dx
        
        for ii,z in enumerate(zip(self.r,self.u,self.e)):
            r = z[0]
            u = z[1]
            e = z[2]
            p = self.p[ii]
            dpde_rc = self.dpde_rc[ii]
            A = self.Geometry.crossSection[ii]
            Ph = self.Geometry.hydraulicPerimeter[ii]
            dQdx = self.dQdx[ii]
            tau_w = self.tau_w[ii]
            dlnA_dx = self.dlnA_dx[ii]
            M = [[u         ,r    ,0        ],
                 [p/(r**2)  ,u    ,1/r*dpde_rc],
                 [0         ,p/r  ,u        ]]
            C = [r*u*dlnA_dx,
                 1/r*tau_w*Ph/A,
                 1/(r*A)*dQdx + p*u/r*dlnA_dx]
            dSV_dt[:,ii] = - np.matmul(M, np.transpose(dSV_dx[:,ii])) - C
        self.dSV_dt = dSV_dt

    def _integrationStep(self,CFL,method='MacCormack'):
        # Adaptive dt for stability using constant CFL
        dt = CFL*np.min((self.Geometry.grid[1:]-self.Geometry.grid[0:-1])/(self.u[1:]+self.a[1:]))
        initial_SV = self.SV
        if method == 'MacCormack':
            # Calculation of the predictor
            self._dSV_dt('forward')
            predictor_dSV_dt = self.dSV_dt
            predictor_SV = initial_SV + predictor_dSV_dt*dt
            # evaluation of SV at predictor values
            self._SV(predictor_SV)
            # Calculation of the corrector from predictor values
            self._dSV_dt('backward')
            corrector_dSV_dt = self.dSV_dt
            # Calculation of the final time derivative (second order accuracy)
            averaged_dSV_dt = 0.5*(predictor_dSV_dt+corrector_dSV_dt)
            final_SV = initial_SV + averaged_dSV_dt*dt
            
            # Applying boundary conditions
            final_SV[0,0]=initial_SV[0,0] # set density
            # final_SV[1,0]=initial_SV[1,0] # set density
            final_SV[1,0]=2*initial_SV[1,1]-initial_SV[1,2] # zero grad conditions
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
        residualSaveStep = 50
        '''
        Performs the integration of the problem according to the method
        specified.
        Methods includes 'MacCormack'
        '''
        residuals = []
        step = 0
        # maximumResidual = np.max(abs(self.SV_residuals[:,1:].flatten()))
        maximumResidual = np.max(abs(self.SV_residuals[2,1:].flatten()))
        while maximumResidual>tol:
            self._integrationStep(CFL, method=method)
            maximumResidual = np.max(self.SV_residuals[:,1:].flatten())
            if np.mod(step,residualSaveStep) == 0:
                residuals.append(np.max(abs(self.SV_residuals[:,1:]),1))
            step += 1
            if showConvergenceProgress:
                print(maximumResidual,
                      np.unravel_index(np.argmax(self.SV_residuals[:,1:]),
                                       self.SV_residuals[:,1:].shape))
            if step == maxSteps:
                break
            if plot and np.mod(step,plotStep)==0:
                if not('fig' in locals()):
                    fig,ax = plt.subplots(10,1,figsize=(6, 10), dpi=400)
                else:
                    plt.rcParams['xtick.labelbottom'] = False
                    ii=0
                    ax[ii].plot(self.Geometry.grid,self.r,label=str(step))
                    ax[ii].set_ylabel('Density [kg/m3]')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.u,label=str(step))
                    ax[ii].set_ylabel('Velocity [m/s]')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.e,label=str(step))
                    ax[ii].set_ylabel('Internal energy [J/kg]')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.p/1e5,label=str(step))
                    ax[ii].set_ylabel('Pressure [bar]')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.T,label=str(step))
                    ax[ii].set_ylabel('Temperature [K]')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.r*self.u*self.A,label=str(step))
                    ax[ii].set_ylabel('Mass flow [kg/s]')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.M,label=str(step))
                    ax[ii].set_ylabel('Mach [-]')
                    ax[ii].tick_params(axis='x',labelbottom='off') # labels along the bottom edge are off
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.SV_residuals[0,:],label=str(step))
                    ax[ii].set_ylabel('Mass equation residuals [-]')
                    ax[ii].set_yscale('log')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.SV_residuals[1,:],label=str(step))
                    ax[ii].set_ylabel('Momentum equation residuals [-]')
                    ax[ii].set_yscale('log')
                    ii=ii+1
                    ax[ii].plot(self.Geometry.grid,self.SV_residuals[2,:],label=str(step))
                    ax[ii].set_ylabel('Energy equation residuals [-]')
                    ax[ii].set_yscale('log')
                    ax[ii].tick_params(axis='x',labelbottom='off') # labels along the bottom edge are off
                    ax[ii].set_xlabel('Axial position [m]')
                    
        if plot:
            plt.show()
            plt.rcParams['xtick.labelbottom'] = True
        status = 'solved'
        if fullOutput:
            sol = {}
            sol['max_residuals'] = np.array(residuals)
            sol['iterations'] = np.linspace(0,step-1,step)
            return(sol)

    def writeResults(self):
        import h5py
    
    def initSonicRocketFlow(self,P0,T0,X0,A,grid,process='quick'):
        from scipy.interpolate import interp1d
        r0 = np.zeros(len(grid))
        e0 = np.zeros(len(grid))
        u0 = np.zeros(len(grid))
        self.gas.TPX = T0, P0, X0
        # inlet
        rSet = np.zeros(3)
        eSet = np.zeros(3)
        uSet = np.zeros(3)
        if process == 'quick':
            throatVelocity,throatDensity,throatPressure = self.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
            massflow = throatVelocity*throatDensity*A[np.argmin(A)]
            velocity,density,e = self.gas.isentropicMassflow(massflow,A[0],throatVelocity,case='subsonic')
            rSet[0] = density
            eSet[0] = e
            uSet[0] = velocity
            self.gas.SP = self.gas.entropy_mass, throatPressure
            e = self.gas.int_energy_mass
            rSet[1] = throatDensity
            eSet[1] = e
            uSet[1] = throatVelocity
            velocity,density,e = self.gas.isentropicMassflow(massflow,A[-1],throatVelocity,case='supersonic',maxMach=5)
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
            throatVelocity,throatDensity,throatPressure = self.gas.chokedNozzle(isentropicEfficiency=1, frozen=True)
            massflow = throatVelocity*throatDensity*A[np.argmin(A)]
            exhaustVelocity,exhaustDensity,e = self.gas.isentropicMassflow(massflow,A[-1],throatVelocity,case='supersonic',maxMach=5)
            maxMach = exhaustVelocity/throatVelocity*1.1
            for ii,z in enumerate(grid):
                u,r,e = self.gas.sonicFlowArea(P0=P0,T0=T0,X0=X0,
                                               throatDensity=throatDensity,
                                               throatVelocity=throatVelocity,
                                               A=A,index=ii,maxMach=maxMach)
                r0[ii] = r
                e0[ii] = e
                u0[ii] = u
        initialTimeState = {'r': r0,'u': u0,'e': e0}
        self.setInitialTimeState(initialTimeState)
        return()
    
def _test__dSV_dx():
    import geometrylib
    # Test of the differentiation method
    # Check that the array method returns the same output as the 
    # loop method where the scheme is more explicit
    nbrPoints = 31
    geometry = geometrylib.OneDGeometry()
    geometry.andersonConstructor(throatDiameter=1,nbrPoints=nbrPoints)
    r00 = 1.2
    rfun = lambda x: r00*(-0.3146*x**2+1)
    grid = geometry.grid
    r0 = rfun(grid)
    initialTimeState = {'r': r0,'u': np.zeros(nbrPoints),'e': np.zeros(nbrPoints)}
    q1DCF = Quasi1DCompressibleFlow(geometry,
                                    fluidModel='perfectgas',
                                    k=1.4,
                                    Mgas = 0.028,
                                    Tw=600,
                                    wallHeatFluxCorrelation='adiabatic', # bartz,adiabatic
                                    wallShearStressCorrelation='non-viscous') # non-viscous, moody_bulk
    q1DCF.setInitialTimeState(initialTimeState)
    q1DCF._SV(q1DCF.SV)
    methods = {'forward':{},
                'backward':{}}
    dx = q1DCF.dx
    for method in methods:
        q1DCF._dSV_dx(method)
        SV_spatial = np.zeros(len(grid))
        SV = q1DCF.r
        for ii,r in enumerate(SV):
            if method=='backward':
                if ii>0:
                    SV_spatial[ii] = (SV[ii] - SV[ii-1])/dx
            if method=='forward':
                if ii < (len(grid)-1):
                    SV_spatial[ii] = (SV[ii+1] - SV[ii])/dx
        methods[method]['array']=q1DCF.dSV_dx[0,:]
        methods[method]['iteration']=SV_spatial
        plt.plot(grid,methods[method]['iteration'],'o',label=method + '_iteration')
        plt.plot(grid,methods[method]['array'],label=method + '_array')
        plt.legend()
    return()

def _test_initSonicRocketFlow():
    import geometrylib
    geometry = geometrylib.OneDGeometry()
    geometry.andersonConstructor(throatDiameter=0.005,nbrPoints=31)
    mech = 'gri30_highT.xml'
    submech = 'gri30_mix'
    fluidModel = 'cantera'
    q1DCF = Quasi1DCompressibleFlow(geometry,
                                    fluidModel=fluidModel,
                                    mech=mech, 
                                    submech=submech,
                                    Tw=600,
                                    wallHeatFluxCorrelation='bartz', # bartz,adiabatic
                                    wallShearStressCorrelation='non-viscous') # non-viscous, moody_bulk
    T0 = 300
    P0 = 25e5
    X0 = 'CH4:1,O2:2'
    q1DCF.gas.TPX = T0,P0,X0
    q1DCF.gas.equilibrate('HP')
    T0 = q1DCF.gas.T
    X0 = q1DCF.gas.X
    q1DCF.initSonicRocketFlow(P0,T0,X0,geometry.crossSection,geometry.grid,process='slow')
    # plt.plot(geometry.grid,q1DCF.h+0.5*(q1DCF.u**2))
    # plt.plot(geometry.grid,q1DCF.p + 0.5*q1DCF.r*(q1DCF.u**2))
       
# _test__dSV_dx()
# _test_initSonicRocketFlow()