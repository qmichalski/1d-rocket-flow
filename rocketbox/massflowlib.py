# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:14:42 2021

@author: Dr Quentin Michalski
        
"""

import cantera as ct
import math
import numpy as np
from scipy.optimize import root, brentq

__version__ = 1.0

class Massflow_Solution(ct.Solution):
    '''
    Inherit the Solution class from Cantera to add the following methods:
        self.HS
        self.HS = H, S
        self.gas_constant
        self.dpde_cr
        self.soundSpeed
        self.chokedNozzle
        
    '''
    def __init__(self, *args):
        ct.Solution.__init__(self, *args)

    @property
    def gas_constant(self):
        return(ct.gas_constant)
    # Defining actions to compute and read a state HS from a Cantera gas object
    
    @property
    def HS(self):
        '''
        Read method for HS state.
            H, S = self.HS
            
        '''
        return (self.enthalpy_mass,self.entropy_mass)
    
    @HS.setter
    def HS(self, val):
        '''
        Write method for HS state similar to standard Cantera calls:
            self.HS = H, S
        Calls the HP method and root find the pressure that yields the
        specified entropy.
            
        '''
        H = val[0]
        S = val[1]
        P = self.P
        self.HP = H,P
        initialGuess = 1.00001
        fun = lambda Pguess: self._HSDiff(Pguess,S)
        # Now rootfind for the pressure that satisfies the H-S inputs provided - note that HSDiff returns the gas to this state (H , P1)
        P1 = brentq(fun,1e2,1e8)
        # Return the gas to its initial state
        self.HP = H,P1
    
    def heatOfReaction(self,reactionType):
        T0,P0,X0 = self.TPX
        h0 = self.enthalpy_mass
        if reactionType=='HP':
            self.equilibrate(reactionType)
        self.TP = T0,P0
        h1 = self.enthalpy_mass
        DrH = h1 - h0
        self.TPX = T0,P0,X0
        return(DrH)
    
    def _HSDiff(self,Pguess,S):
        '''
        Intermediary function that is then called in root finder
        to zero on entropy.
            
        '''
        # Store gas current state
        P = self.P
        H = self.enthalpy_mass
        # Update to guessed pressure at constant enthalpy
        self.HP = H,Pguess
        # Determine the difference between specified S and s(h,Pguess)
        error = self.entropy_mass-S
        # Return gas to initial state
        self.HP = H,P
        return error

    def dpde_cr(self):
        # save properties
        e0 = self.int_energy_mass
        p0 = self.P
        r0 = self.density
        e1 = e0*1.0001
        self.UV = e1, 1/r0
        p1 = self.P
        # drhodP_cH = (r1 - r0)/(p1 - p0)
        dpde_cr_val = (p1-p0)/(e1-e0)
        # Return the gas object to its original state 
        self.UV = e0, 1/r0
        return dpde_cr_val

    def soundSpeed(self,frozen=True):
        """
        Returns the frozen sound speed for a gas by using a finite-difference 
        approximation of the derivative.
        If frozen is at True, compute chemically frozen sound speed.
        If frozen is at False, compute sound speed at chemical equilibrium.
        """
        # Save properties
        s0 = self.s
        p0 = self.P
        r0 = self.density
        # Perturb the pressure
        p1 = p0*1.0001
        # set the gas to a state with the same entropy and composition but
        # the perturbed pressure
        self.SP = s0, p1
        if not(frozen):
            # if not frozen, chemistry is at equilibrium
            self.equilibrate('SP')
        # frozen sound speed
        a = math.sqrt((p1 - p0)/(self.density - r0))
        # Return the gas object to its original state 
        self.SP = s0,p0
        return a
        
    def _diffChoking(self,throatPressure,isentropicEfficiency=1,frozen=True):
        """
        Returns the difference between the sound speed of a gas and 
        the velocity predicted by the adiabatic one-dimensional 
        energy equation.
        Can be called by a root-finding algorithm to drive this difference 
        to zero (and hence find the conditions at which the velocity 
        is equal to the speed of sound)
        """
        # Evaluate two independent intensive properties of the gas (so we can reset it to its initial state before exiting the function)
        initialEntropy,stagnationPressure = self.SP
        # Evaluate the stagnation enthalpy of the gas
        stagnationEnthalpy = self.enthalpy_mass
        # Update the gas to the estimated throat pressure (isentropic expansion to throatPressure)
        self.SP = self.entropy_mass,throatPressure
        # From the definition of the isentropic efficiency:
        h2s = self.enthalpy_mass
        # Throat enthalpy is higher in the irreversible expansion to the same final pressure
        h2 = stagnationEnthalpy - isentropicEfficiency*(stagnationEnthalpy-h2s)
        # Calculate the specific kinetic energy achieved during this expansion
        specificKineticEnergy = stagnationEnthalpy-h2
        # Velocity from specific kinetic energy
        velocity = math.sqrt(2*specificKineticEnergy)
        # Find the sound speed of the gas at this condition
        soundSpeed = self.soundSpeed(frozen=frozen)
        # Return the gas to its initial conditions
        self.SP = initialEntropy,stagnationPressure
        # The error term for choking is the difference between the velocity at the end of the expansion and the sound speed at the end of the expansion
        error = soundSpeed-velocity
        return error

    # Adiabatic irreversible choked nozzle #
    def chokedNozzle(self,isentropicEfficiency=1,frozen=True):
        """
        Compute the choked state on a gas object (assumes it is choked).
        Returns velocity, density and pressure at throat.
        """
        # Save properties
        s0,P0 = self.SP
        # Stagnation enthalpy
        stagnationEnthalpy = self.enthalpy_mass
        # Find the throat pressure which results in choked flow for this irreversible nozzle
        fun = lambda throatPressure: self._diffChoking(throatPressure,isentropicEfficiency=1,frozen=frozen)
        sol = root(fun,[self.P*0.9])
        throatPressure=sol['x'][0]
        self.SP = self.entropy_mass,throatPressure
        # From the definition of the isentropic efficiency:
        h2s = self.enthalpy_mass
        # Throat enthalpy is higher in the irreversible expansion to the same final pressure
        h2 = stagnationEnthalpy - isentropicEfficiency*(stagnationEnthalpy-h2s)
        # Calculate the specific kinetic energy achieved during this expansion
        specificKineticEnergy = stagnationEnthalpy-h2
        # Velocity from specific kinetic energy
        velocity = math.sqrt(2*specificKineticEnergy)
        # Throat velocity from adiabatic one-dimensional energy equation
        velocity = math.sqrt(2*(stagnationEnthalpy-self.enthalpy_mass))
        density = self.density
        # Update the gas object to the stagnation conditions
        self.SP = s0,P0
        return velocity,density,throatPressure
    
    def _diffIsentropicMassFlux(self,u,massflow,A,fullOutput=False): 
        # print(u,A)
        s0 = self.entropy_mass
        h0 = self.enthalpy_mass
        h = h0 - 0.5*(u**2)
        self.HS = h, s0
        density = self.density
        e = self.int_energy_mass
        diff = (u*density*A - massflow)/massflow
        self.HS = h0, s0
        if fullOutput:
            return u,density,e
        else:
            return(diff)

    def isentropicMassflow(self,massflow,A,throatVelocity,case,maxMach=5,debugging=False):
        from scipy.optimize import brentq
        fun = lambda u: self._diffIsentropicMassFlux(u,massflow, A)
        if case =='subsonic':
            u = brentq(fun,0,throatVelocity)
        if case =='sonic':
            u = throatVelocity
        if case =='supersonic':
            found = False
            uMax = maxMach*throatVelocity
            T,P,X = self.TPX
            while not(found):
                try:
                    u = brentq(fun,throatVelocity,uMax)
                    found = True
                except Exception as e:
                    self.TPX = T,P,X
                    if debugging:
                        print(e)
                    if uMax <= throatVelocity:
                        found = True
                    else: 
                        uMax = uMax*0.99
        velocity,density,e = self._diffIsentropicMassFlux(u,massflow,A,fullOutput=True)
        return(velocity,density,e)

    def sonicFlowArea(self,P0,T0,X0,throatDensity,throatVelocity,A,index,maxMach=5):
        '''
        Initialization of flow field when smallest area in the engine is choked
        '''
        self.TPX = T0, P0, X0
        A_throat = A[np.argmin(A)]
        massflow = throatVelocity*throatDensity*A_throat
        if index < np.argmin(A):
            case = 'subsonic'
        if index == np.argmin(A):
            case = 'sonic'
        if index > np.argmin(A):
            case = 'supersonic'
        velocity,density,e = self.isentropicMassflow(massflow,A[index],throatVelocity,case,maxMach=maxMach)
        # print(index,velocity,density)
        return(velocity,density,e)
    
def _test_soundSpeed():
    gas = Massflow_Solution('gri30.cti')
    gas.TPX = 300,1e5,'N2:1'
    a = gas.soundSpeed()
    print('Nitrogen (at {} K and {} bar) sound speed is {} m/s'.format(gas.T,gas.P/1e5,a))
    return()

def _test_chokedNozzle():
    import numpy as np
    gas = Massflow_Solution('gri30.cti')
    gas.TPX = 300,10e5,'N2:1'
    a = gas.soundSpeed()
    v,r,p = gas.chokedNozzle()
    area = np.pi*(10/2*1e-3)**2
    mdot = v*r*area
    print('Throat mass flux of nitrogen (at {} K and {} bar) is {} kg/s/m2'.format(gas.T,gas.P/1e5,v*r))
    print('Mass flow for a 10 mm throat is {} kg/s'.format(mdot))
    return()

def _test_HS():
    gas = Massflow_Solution('gri30.cti')
    for T0 in [300,500,1000,2000,3000]:
        P0 = 1e5
        X0 = 'N2:1'
        gas.TPX = T0, P0, X0
        H = gas.enthalpy_mass
        S = gas.entropy_mass
        # print(H,S)
        for p in [1e5,10e5,100e5,1000e5]:
            T1 = 1000
            P1 = 10e5
            gas.TP = T1, P1
            try:
                gas.HS = H, S
                print('T0 {} K, P0 {} bar: dT/T0 {} and dP/P0 {} by HS'.format(T0, P0, abs(gas.T-T0)/T0,abs(gas.P-P0)/P0))
            except:
                print('Moving from T1 {} K to T0 {} K, P1 {} bar to P0 {} bar'.format(T1,T0,P1,P0))
    
def _test_diffIsentropicMassFlux():
    from matplotlib import pyplot as plt
    gas = Massflow_Solution('gri30.cti')
    gas.TPX = 2000,10e5,'N2:1'
    u,r,p = gas.chokedNozzle()
    print('Throat velocity {} m/s, density {} kg/m3'.format(u,r))
    A_throat = 1
    massflow = u*r*A_throat
    uSet = np.linspace(0, 2000,200)
    diffSet = np.zeros(len(uSet))
    for ii,u in enumerate(uSet):
        diffSet[ii] = gas._diffIsentropicMassFlux(u,massflow,A_throat*2,fullOutput=False)
    plt.plot(uSet,diffSet)
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Mass flow - mass flow calculated at velocity [kg/s]')
    
def _test_isentropicMassflow():
    gas = Massflow_Solution('gri30.cti')
    gas.TPX = 2000,10e5,'N2:1'
    throatVelocity,r,p = gas.chokedNozzle()
    A_throat = 1
    massflow = throatVelocity*r*A_throat
    u,r,e = gas.isentropicMassflow(massflow,A_throat*2,throatVelocity,case='subsonic')
    print('Subsonic solution: Velocity {} m/s, density {} kg/m3, massflow difference {} kg/s'.format(u,r,(u*r*A_throat*2-massflow)/massflow))
    u,r,e = gas.isentropicMassflow(massflow,A_throat*2,throatVelocity,case='supersonic',maxMach=5)
    print('Supersonic solution: Velocity {} m/s, density {} kg/m3, massflow difference {} kg/s'.format(u,r,(u*r*A_throat*2-massflow)/massflow))
    
def _test_sonicFlowArea():
    from matplotlib import pyplot as plt
    from time import time
    gas = Massflow_Solution('gri30.cti')
    throatDiameter = 0.005
    Afun = lambda x: throatDiameter*(1 + 2.2*(x-1.5)**2)
    length = 3
    grid = np.linspace(0,length,31)
    A = Afun(grid)
    P0 = 200e5
    T0 = 3000
    X0 ='N2:1'
    gas.TPX = T0,P0,X0
    # gas.equilibrate('HP')
    T0 = gas.T
    X0 = gas.X
    throatVelocity,throatDensity,p = gas.chokedNozzle()
    results = {
        'massflow':np.zeros(len(grid)),
        'T':np.zeros(len(grid)),
        'runTime':np.zeros(len(grid)),
        'density':np.zeros(len(grid)),
        'P':np.zeros(len(grid)),
        'H':np.zeros(len(grid)),
        'U':np.zeros(len(grid)),
        'Ht':np.zeros(len(grid)),
        'Pt':np.zeros(len(grid)),
        'S':np.zeros(len(grid)),
        'A':np.zeros(len(grid)),
        'Mach':np.zeros(len(grid))
        }
    massflow = throatVelocity*throatDensity*A[np.argmin(A)]
    exhaustVelocity,exhaustDensity,e = gas.isentropicMassflow(massflow,A[-1],throatVelocity,case='supersonic',maxMach=5)
    maxMach = exhaustVelocity/throatVelocity*1.1
    # print(throatVelocity,throatDensity,massflow,maxMach)
    for ii,x in enumerate(grid):
        t1 = time()
        u,r,e = gas.sonicFlowArea(P0=P0,T0=T0,X0=X0,
                                  throatDensity=throatDensity,
                                  throatVelocity=throatVelocity,
                                  A=A,index=ii,maxMach=maxMach)
        t2 = time()
        results['runTime'][ii] = t2 - t1
        gas.UV = e,1/r
        results['massflow'][ii] = u*r*A[ii]
        results['T'][ii] = gas.T
        results['density'][ii] = gas.density
        results['P'][ii] = gas.P
        results['H'][ii] = gas.enthalpy_mass
        results['U'][ii] = u
        results['Ht'][ii] = results['H'][ii]+0.5*(results['U'][ii]**2)
        results['S'][ii] = gas.entropy_mass
        results['A'][ii] = A[ii]
        results['Mach'][ii] = u/gas.soundSpeed(frozen=True)
        gas.HS = results['Ht'][ii],results['S'][ii]
        # results['Pt'][ii] = results['P'][ii]+0.5*results['density'][ii]*(results['U'][ii]**2)
        results['Pt'][ii] = gas.P
        
    plt.plot(grid,(results['massflow']-results['massflow'][0])/results['massflow'][0])
    plt.ylabel('Massflow relative to massflow inlet [-]')
    plt.xlabel('Position [m]')
    plt.show()
    plt.plot(grid,results['density'])
    plt.ylabel('Density [kg/m3]')
    plt.xlabel('Position [m]')
    plt.show()
    plt.plot(grid,results['T'])
    plt.ylabel('Temperature [K]')
    plt.xlabel('Position [m]')
    plt.show()
    plt.plot(grid,results['P']/1e5)
    plt.ylabel('Pressure [bar]')
    plt.xlabel('Position [m]')
    plt.show()
    plt.plot(grid,results['Ht'])
    plt.ylabel('Total enthalpy [J/kg]')
    plt.xlabel('Position [m]')
    plt.show()
    plt.plot(grid,results['Pt']/1e5)
    plt.ylabel('Total pressure [bar]')
    plt.xlabel('Position [m]')
    plt.show()
    plt.plot(grid,results['S'])
    plt.ylabel('Entropy [J/kg/K]')
    plt.xlabel('Position [m]')
    plt.show()
    plt.plot(grid,results['runTime']*1e3,'o-')
    plt.ylabel('Point run time [ms]')
    plt.xlabel('Position [m]')
    plt.show()

    return(results)

# _test_diffIsentropicMassFlux()
# _test_isentropicMassflow()
# results = _test_sonicFlowArea()
# _test_soundSpeed()
# _test_chokedNozzle()
# _test_HS()
