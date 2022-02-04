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
                                innerCore=False,
                                innerCoreDict=None,
                                nbrPoints=500,
                                plot=False):
        
        from scipy.optimize import root
        import numpy as np
        import matplotlib.pyplot as plt
        innerCoreDiameter = 0
        
        LC = combustionChamberLength
        Rt = throatDiameter/2
        RC = chamberDiameter/2
        Rn = nozzleDiameter/2
        Rconge = 0
        
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
        
        xSet = np.linspace(0,combustionChamberLength+nozzleLength,nbrPoints)
        ySet = np.zeros(len(xSet))
        
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
        
        ySet_inner = np.zeros(len(xSet))
        
        if innerCore == True:
            if innerCoreDict['type'] == 'convergent':
                print(innerCoreDict['type'])
                xFinal = RC / np.tan(innerCoreDict['innerAngle']) + innerCoreDict['flatLength']
                xM = innerCoreDict['flatLength']
                if xM > xA:
                    xM = xA
                for ii,x in enumerate(xSet):
                    if x <= xM:
                        ySet_inner[ii] = RC - innerCoreDict['channelWidth']
                    if xM < x <= xFinal:
                        ySet_inner[ii] = RC - innerCoreDict['channelWidth'] - np.tan(innerCoreDict['innerAngle'])*(x-xM)
                S_outer = np.pi*ySet**2
                S_tot = np.pi*(ySet**2-ySet_inner**2)
                # if xSet[-1] > xFinal:
                argXFinal = np.where(xSet>xFinal)[0][0]
                dS = S_outer - S_tot
                # print(np.argmin(dS[0:argXFinal]))
                for ii,x in enumerate(xSet):
                    if xSet[np.argmin(dS[0:argXFinal])] <= x:
                            ySet_inner[ii] = 0
                            
            if innerCoreDict['type'] == 'copy':
                print(innerCoreDict['type'])
                ySet_inner = ySet - innerCoreDict['channelWidth']
                ySet_inner[xSet>(xK-25e-3)] = 0
                
                # for ii,x in enumerate(xSet):
                #     if (xK) < x <= combustionChamberLength:
                #         ySet[ii] = (1+1.5)*Rt - ((1.5*Rt)**2 - (LC-x)**2)**(1/2)
                
        # model properties
        self._gridLength = xSet[-1]
        self._chamberDiameter = chamberDiameter
        self._convergentAngle = convergentAngle
        self._throatDiameter = throatDiameter
        self._nbrPoints = nbrPoints
        # essential properties
        self.grid = xSet
        self.crossSection = np.pi*(ySet**2 - ySet_inner**2)
        self.hydraulicPerimeter = 2*np.pi*(ySet + ySet_inner)
        self.hydraulicDiameter = 4*self.crossSection/self.hydraulicPerimeter
        self.heatExchangePerimeter = self.hydraulicPerimeter
        self.roughness = roughness*np.ones(len(xSet))
        
        if plot:
            RC = np.max(ySet)
            plt.figure(figsize=(6, 6*5*RC/(LC*1.2)), dpi=400)
            plt.plot(self.grid,ySet,'-x',color='k')
            plt.plot(self.grid,-ySet,'-x',color='k')
            if innerCore == True:
                plt.plot(self.grid,ySet_inner,'-x',color='b')
                plt.plot(self.grid,-ySet_inner,'-x',color='b')
            plt.xlim([0,self.grid[-1]*1.2])
            plt.ylim([-RC*1.2,RC*1.2])
            plt.show()
            
            plt.plot(self.crossSection)

    def rdeEngineConstructor(self,
                            outerFlatLength,
                            chamberDiameter,
                            convergentAngle,
                            throatDiameter,
                            roughness,
                            nozzleLength,
                            nozzleDiameter,
                            innerCore=False,
                            innerCoreDict=None,
                            nbrPoints=500,
                            plot=False):
    
        from scipy.optimize import root
        import numpy as np
        import matplotlib.pyplot as plt
        innerCoreDiameter = 0
        
        Rt = throatDiameter/2
        RC = chamberDiameter/2
        Rn = nozzleDiameter/2
        Rconge = 0
        
        convergentAngleMax = np.arctan(((RC-2.5*Rt)**2/((1.5*Rt)**2-(RC-2.5*Rt)**2))**(1/2))/np.pi*180
        convergentAngleRad = convergentAngle / 180 * np.pi
        if convergentAngleRad > convergentAngleMax and convergentAngleMax > 0:
            print('convergent Angle over convergent Angle max = {} - consider smaller angle'.format(convergentAngleMax*180/np.pi))
        
        yF = Rt - 1.5*Rt*(1/np.cos(convergentAngleRad)-1)
        
        combustionChamberLength = (RC-yF) / np.tan(convergentAngleRad) + outerFlatLength
        LC = combustionChamberLength
        # print(yF,RC,Rt,LC)
        
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
        
        xSet = np.linspace(0,combustionChamberLength+nozzleLength,nbrPoints)
        ySet = np.zeros(len(xSet))

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

        ySet_inner = np.zeros(len(xSet))
        
        if innerCore == True:
            if innerCoreDict['type'] == 'convergent':
                RCi = RC - innerCoreDict['channelWidth']
                yDi = RCi - Rconge
                
                print(innerCoreDict['type'])
                xFinal = (RC-innerCoreDict['channelWidth']) / np.tan(innerCoreDict['innerAngle']) + innerCoreDict['flatLength']
                xM = innerCoreDict['flatLength']
                if xM > xA:
                    xM = xA
                for ii,x in enumerate(xSet):
                    if x <= xM:
                        ySet_inner[ii] = RC - innerCoreDict['channelWidth']
                    if xM < x <= xFinal:
                        ySet_inner[ii] = RC - innerCoreDict['channelWidth'] - np.tan(innerCoreDict['innerAngle'])*(x-xM)
                S_outer = np.pi*ySet**2
                S_tot = np.pi*(ySet**2-ySet_inner**2)
                # if xSet[-1] > xFinal:
                argXFinal = np.where(xSet>xFinal)[0][0]
                dS = S_outer - S_tot
                # print(np.argmin(dS[0:argXFinal]))
                for ii,x in enumerate(xSet):
                    if xSet[np.argmin(dS[0:argXFinal])] <= x:
                            ySet_inner[ii] = 0
                            
            if innerCoreDict['type'] == 'copy':
                print(innerCoreDict['type'])
                ySet_inner = ySet - innerCoreDict['channelWidth']
                ySet_inner[xSet>(xK-25e-3)] = 0
                
                # for ii,x in enumerate(xSet):
                #     if (xK) < x <= combustionChamberLength:
                #         ySet[ii] = (1+1.5)*Rt - ((1.5*Rt)**2 - (LC-x)**2)**(1/2)
                
        # model properties
        self._gridLength = xSet[-1]
        self._chamberDiameter = chamberDiameter
        self._convergentAngle = convergentAngle
        self._throatDiameter = throatDiameter
        self._nbrPoints = nbrPoints
        # essential properties
        self.grid = xSet
        self.crossSection = np.pi*(ySet**2 - ySet_inner**2)
        self.hydraulicPerimeter = 2*np.pi*(ySet + ySet_inner)
        self.hydraulicDiameter = 4*self.crossSection/self.hydraulicPerimeter
        self.heatExchangePerimeter = self.hydraulicPerimeter
        self.roughness = roughness*np.ones(len(xSet))
        
        if plot:
            RC = np.max(ySet)
            plt.figure(figsize=(6, 6*5*RC/(LC*1.2)), dpi=400)
            plt.plot(self.grid,ySet,'-',color='k')
            plt.plot(self.grid,-ySet,'-',color='k')
            plt.plot(self.grid,ySet_inner,'-x',color='b')
            plt.plot(self.grid,-ySet_inner,'-x',color='b')
            
            plt.xlim([0,self.grid[-1]*1.2])
            # plt.xlim([0.025,0.05])
            plt.ylim([-RC*1.2,RC*1.2])
            # plt.plot(xA,yA,'o',label='A')
            # plt.plot(xB,yB,'o',label='B')
            # plt.plot(xC,yC,'o',label='C')
            # plt.plot(xD,yD,'o',label='D')
            # plt.legend()
            plt.show()
            
            plt.plot(self.crossSection)

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
    import numpy as np
    geometry = OneDGeometry()
    # geometry.rocketEngineConstructor(combustionChamberLength=100e-3,
    #                                  chamberDiameter=50e-3,
    #                                  convergentAngle=15,
    #                                  throatDiameter=25e-3,
    #                                  roughness=10e-6,
    #                                  nozzleLength=50e-3,
    #                                  nozzleDiameter=50e-3,
    #                                  plot=True,nbrPoints=50)
    innerCoreDict = {
        'type':'convergent',
        'channelWidth':5e-3,
        'flatLength':30e-3,
        'innerAngle':21.8/180*np.pi
        }
    
    # geometry.rocketEngineConstructor(combustionChamberLength=100e-3,
    #                                  chamberDiameter=50e-3,
    #                                  convergentAngle=15,
    #                                  throatDiameter=25e-3,
    #                                  roughness=10e-6,
    #                                  nozzleLength=15e-3,
    #                                  nozzleDiameter=30e-3,
    #                                  innerCore=True,
    #                                  innerCoreDict=innerCoreDict,
    #                                  plot=True,nbrPoints=100)
    
    geometry.rdeEngineConstructor(outerFlatLength=innerCoreDict['flatLength'],
                                  chamberDiameter=100e-3,
                                  convergentAngle=15,
                                  throatDiameter=30.8e-3,
                                  roughness=10e-6,
                                  nozzleLength=15e-3,
                                  nozzleDiameter=35e-3,
                                  innerCore=True,
                                  innerCoreDict=innerCoreDict,
                                  plot=True,nbrPoints=100)
    
    # geometry.andersonConstructor(plot=True)
    return()

if __name__ == "__main__":

    _test_rocketEngineConstructor()
