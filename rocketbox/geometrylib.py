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
                                nozzleLength,
                                nozzleDiameter,
                                nbrPoints=500,
                                nozzleType='bellApproximation',plot=False):
        
        from scipy.optimize import root
        import numpy as np
        import matplotlib.pyplot as plt
        innerCoreDiameter = 0
        xSet = np.linspace(0,combustionChamberLength+nozzleLength,nbrPoints)
        ySet = np.zeros(len(xSet))
        
        LC = combustionChamberLength
        Rt = throatDiameter/2
        RC = chamberDiameter/2
        Rn = nozzleDiameter/2
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
        Rthroatnozzle = 1.5*Rt
        fun = lambda theta: np.tan(theta)-(Rn-(Rt+Rthroatnozzle*(1-np.cos(theta))))/(nozzleLength-Rthroatnozzle*np.sin(theta))
        thetaN = root(fun,0)['x'][0]
        # thetaN = 30/180*np.pi
        # plt.plot(np.linspace(0,45,100),fun(np.linspace(0,np.pi/4,100)))
        xN = Rthroatnozzle*np.sin(thetaN) + LC
        yN = Rthroatnozzle*(1-np.cos(thetaN)) + Rt
        # print(thetaN*180/np.pi,xN,yN)
        for ii,x in enumerate(xSet):
            if x <= xA:
                ySet[ii] = RC
                # print(0,x)
            if xA < x <= xB:
                ySet[ii] = yD + ((Rconge)**2 - (xD-x)**2)**(1/2)
            if xB < x <= xK:
                ySet[ii] = RC - np.tan(convergentAngleRad)*(x-(xK-dxC))
            if (xK) < x <= combustionChamberLength:
                ySet[ii] = (1+1.5)*Rt - ((1.5*Rt)**2 - (LC-x)**2)**(1/2)
            if LC < x <= xN:
                ySet[ii] = Rt+Rthroatnozzle - (Rthroatnozzle**2 - (LC-x)**2)**(1/2)
            if xN < x:
                ySet[ii] = yN + np.tan(thetaN)*(x-xN)
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
            RC = np.max(self.hydraulicDiameter)/2
            plt.figure(figsize=(6, 6*5*RC/(LC*1.2)), dpi=400)
            plt.plot(self.grid,self.hydraulicDiameter/2,'-',color='k')
            plt.plot(self.grid,-self.hydraulicDiameter/2,'-',color='k')
            plt.xlim([0,self.grid[-1]*1.2])
            plt.ylim([-RC*1.2,RC*1.2])
            plt.show()
        
    def andersonConstructor(self, throatDiameter = 1, nbrPoints=51, roughness=10e-6, plot=False):
        import numpy as np
        import matplotlib.pyplot as plt
        Afun = lambda x: throatDiameter*(1 + 2.2*(x-1.5)**2)
        length = 3
        grid = np.linspace(0,length,nbrPoints)
        self.grid = grid
        self.crossSection = Afun(grid)
        self.hydraulicDiameter = (4*Afun(grid)/np.pi)**(1/2)
        self.hydraulicPerimeter = 2*np.pi*self.hydraulicDiameter
        self.heatExchangePerimeter = self.hydraulicPerimeter
        self.roughness = roughness*np.ones(nbrPoints)
        if plot:
            RC = np.max(self.hydraulicDiameter)/2
            plt.figure(figsize=(6, 6*5*RC/(length*1.2)), dpi=400)
            plt.plot(self.grid,self.hydraulicDiameter/2,color='k')
            plt.plot(self.grid,-self.hydraulicDiameter/2,color='k')
            plt.xlim([0,length*1.2])
            plt.ylim([-RC*2.5,RC*2.5])
        
def _test_rocketEngineConstructor():
    geometry = OneDGeometry()
    geometry.rocketEngineConstructor(combustionChamberLength=100e-3,
                                     chamberDiameter=50e-3,
                                     convergentAngle=25,
                                     throatDiameter=25e-3,
                                     roughness=10e-6,
                                     nozzleLength=50e-3,
                                     nozzleDiameter=50e-3,
                                     plot=True,nbrPoints=500)
    
    # geometry.andersonConstructor(plot=True)
    return()

# _test_rocketEngineConstructor()
