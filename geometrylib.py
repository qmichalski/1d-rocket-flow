# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:09:49 2021

@author: quent
"""

class EngineGeometry():
    
def engineConstructor(combustionChamberLength,chamberDiameter,convergentAngle,throatDiameter,nbrPoints=500,nozzleType='bellApproximation',plot=False):
    from scipy.optimize import root
    innerCoreDiameter = 0
    xSet = np.linspace(0,combustionChamberLength+10e-3,nbrPoints)
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
    
    # throatIndexStart = np.where(xSet<(combustionChamberLength-Rt*5e-1))[0][-1]
    # throatIndexEnd = np.where(xSet>(combustionChamberLength+Rt*5e-1))[0][0]
    # print(throatIndexStart,throatIndexEnd)
    # ySet[throatIndexStart:throatIndexEnd] = Rt 
    
    geometryDict = {}
    geometryDict['channelLength'] = lambda x: xSet[-1]
    geometryDict['chamberDiameter'] = lambda x: chamberDiameter
    geometryDict['convergentAngle'] = lambda x: convergentAngle
    geometryDict['throatDiameter'] = lambda x: throatDiameter
    geometryDict['nbrPoints'] = lambda x: nbrPoints
    channelRadiusFun = interp1d(xSet,ySet,kind='linear')
    geometryDict['channelRadius'] = lambda x: channelRadiusFun(x)
    
    if innerCoreDiameter == 0:
        channelSectionSet = np.pi*ySet**2
        channelSectionDerivativeSet[1:] = np.diff(channelSectionSet) / np.diff(xSet)
        channelPerimeterSet = 2*np.pi*ySet
    
    channelSectionFun = interp1d(xSet,channelSectionSet,kind='linear')
    geometryDict['channelSection'] = lambda x: channelSectionFun(x)
    channelSectionDerivativeFun = interp1d(xSet,channelSectionDerivativeSet,kind='linear')
    geometryDict['channelSectionDerivative'] = lambda x: channelSectionDerivativeFun(x)
    channelPerimeterFun = interp1d(xSet,channelPerimeterSet,kind='linear')
    geometryDict['channelPerimeter'] = lambda x: channelPerimeterFun(x)
    geometryDict['diameterHydro'] = lambda x: 4 * geometryDict['channelSection'](x) / geometryDict['channelPerimeter'](x)
    geometryDict['roughness'] = lambda x: 10e-6
    geometryDict['exchangePerimeter'] = lambda x: channelPerimeterFun(x)
   
    if plot:
        plt.figure(figsize=(6, 6*5*RC/(LC*1.2)), dpi=400)
        # xCircle = LC+1.5*Rt*np.cos(np.linspace(0,2*np.pi,50))
        # yCircle = 2.5*Rt+1.5*Rt*np.sin(np.linspace(0,2*np.pi,50))
        # plt.plot(xCircle,yCircle,color='red')
        
        plt.plot(xSet,ySet,color='k')
        plt.plot(xSet,-ySet,color='k')
        plt.xlim([0,LC*1.2])
        plt.ylim([-RC*2.5,RC*2.5])
        x=0.08200411443563355
        plt.plot([x],geometryDict['channelRadius'](x),'x')
        # plt.plot(xK,yK,'x')
        # plt.plot(xC,yC,'x')
        # plt.plot(xD,yD,'x')
        # plt.plot(xB,yB,'x')
        # plt.plot(xA,yA,'x')
        
        plt.show()
        
        plt.plot(xSet,channelSectionSet)
        plt.show()
        
        plt.plot(xSet,channelSectionDerivativeSet,color='k')
        
        plt.show()
        
    return(geometryDict)