# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:09:49 2021

@author: quent
"""

class OneDGeometry():
    def __init__(
        self, grid = None, A = None, Ph = None, Dh = None, Pexch = None,
        roughness = None):
        if not(grid==None):
            self.grid = grid
        if not(A==None):
            self.crossSection = A
        if not(Ph==None):
            self.hydraulicPerimeter = Ph
        if not(Dh==None):
            self.hydraulicDiameter = Dh
        if not(Pexch==None):
            self.heatExchangePerimeter = Pexch
        if not(roughness==None):
            self.roughness = roughness
        self.requiredProperties = ['grid',
                                   'crossSection',
                                   'hydraulicPerimeter',
                                   'hydraulicDiameter',
                                   'heatExchangePerimeter',
                                   'roughness']
 
    def rocketEngineConstructor(self,
                                combustionChamberLength,
                                chamberDiameter,
                                convergentAngle,
                                throatDiameter,
                                roughness,
                                nbrPoints=500,
                                nozzleType='bellApproximation',plot=False):
        
        from scipy.optimize import root
        import numpy as np
        import matplotlib.pyplot as plt
        innerCoreDiameter = 0
        xSet = np.linspace(0,combustionChamberLength,nbrPoints)
        ySet = np.zeros(len(xSet))
        channelSectionSet = np.zeros(len(xSet))
        channelSectionDerivativeSet = np.zeros(len(xSet))
        channelPerimeterSet = np.zeros(len(xSet))
        
        LC = combustionChamberLength
        Rt = throatDiameter/2
        RC = chamberDiameter/2
        Rconge = Rt
        
        convergentAngleMax = np.arctan(((RC-2.5*Rt)**2/((1.5*Rt)**2-(RC-2.5*Rt)**2))**(1/2))/np.pi*180
        convergentAngleRad = convergentAngle / 180 * np.pi
        if convergentAngleRad > convergentAngleMax and convergentAngleMax > 0:
            print('convergent Angle over convergent Angle max = {} - consider smaller angle'.format(convergentAngleMax*180/np.pi))
        xK =  LC - 1.5*Rt*np.cos(np.pi/2 - convergentAngleRad)
        yK = 2.5*Rt - ((1.5*Rt)**2 - (xK-LC)**2)**(1/2)
        dxC = (RC-yK) / np.tan(convergentAngleRad)
        xC = xK-dxC
        yC = RC
        yD = RC - Rconge
        yB = yD + Rconge*np.sin(np.pi/2-convergentAngleRad)
        fun = lambda x: (x-xC)**2 / (np.sin(np.pi/2-convergentAngleRad)**2) - (xC - x + Rconge*np.cos(np.pi/2-convergentAngleRad))**2 - (yC-yD)**2 + Rconge**2
        sol = root(fun,xC)
        xB = sol['x'][0]
        xD = xB - Rconge*np.cos(np.pi/2-convergentAngleRad)
        xA = xD
        yA = RC
        
        for ii,x in enumerate(xSet):
            if x <= xA:
                ySet[ii] = RC
                # print(0,x)
            if xA < x <= xB:
                ySet[ii] = yD + ((Rconge)**2 - (xD-x)**2)**(1/2)
            if xB < x <= xK:
                ySet[ii] = RC - np.tan(convergentAngleRad)*(x-(xK-dxC))
            if (xK) < x <= combustionChamberLength:
                ySet[ii] = 2.5*Rt - ((1.5*Rt)**2 - (LC-x)**2)**(1/2)
            if LC < x:
                ySet[ii] = 2.5*Rt - ((1.5*Rt)**2 - (LC-x)**2)**(1/2)
        
        # model properties
        self._gridLength = xSet[-1]
        self._chamberDiameter = chamberDiameter
        self._convergentAngle = convergentAngle
        self._throatDiameter = throatDiameter
        self._nbrPoints = nbrPoints
        # essential properties
        self.grid = xSet
        self.crossSection = np.pi*ySet**2
        self.hydraulicPerimeter = 2*np.pi*ySet
        self.hydraulicDiameter = 4*self.crossSection/self.hydraulicPerimeter
        self.heatExchangePerimeter = self.hydraulicPerimeter
        self.roughness = roughness*np.ones(len(xSet))
        
        if plot:
            plt.figure(figsize=(6, 6*5*RC/(LC*1.2)), dpi=400)
            plt.plot(self.grid,self.hydraulicDiameter,color='k')
            plt.plot(self.grid,-self.hydraulicDiameter,color='k')
            plt.xlim([0,LC*1.2])
            plt.ylim([-RC*2.5,RC*2.5])
            
            plt.show()
        
    def andersonConstructor(self,nbrPoint=51, roughness=10e-6, plot=False):
        import numpy as np
        import matplotlib.pyplot as plt
        Afun = lambda x: 1 + 2.2*(x-1.5)**2
        length = 3
        grid = np.linspace(0,length,nbrPoint)
        self.grid = grid
        self.crossSection = Afun(grid)
        self.hydraulicDiameter = (4*Afun(grid)/np.pi)**(1/2)
        self.hydraulicPerimeter = 2*np.pi*self.hydraulicDiameter
        self.heatExchangePerimeter = self.hydraulicPerimeter
        self.roughness = roughness*np.ones(nbrPoint)
        if plot:
            RC = np.max(self.hydraulicDiameter)
            plt.figure(figsize=(6, 6*5*RC/(length*1.2)), dpi=400)
            plt.plot(self.grid,self.hydraulicDiameter,color='k')
            plt.plot(self.grid,-self.hydraulicDiameter,color='k')
            plt.xlim([0,length*1.2])
            plt.ylim([-RC*2.5,RC*2.5])
        
def _test_rocketEngineConstructor():
    geometry = OneDGeometry()
    geometry.rocketEngineConstructor(100e-3,50e-3,45,25e-3,10e-6,plot=True)
    geometry.andersonConstructor(plot=True)
    return()

# _test_rocketEngineConstructor()
