"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""


import numpy as np
import skfuzzy.control as ctrl
import FuzzyRules as fr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


class FuzzyControl():
    
    
    ############################################################    
    def __init__(self): 
                
        # Sparse universe
        self.universe = np.linspace(-2, 2, 5)
        

        # fuzzy variables
        self.GPA = ctrl.Antecedent(self.universe, 'GPA')
        self.GPAchanging = ctrl.Antecedent(self.universe, 'GPAchanging')
        self.RangeOfCourses = ctrl.Antecedent(self.universe, 'RangeOfCourses')
        self.typeOfGrade = ctrl.Antecedent(self.universe, 'typeOfGrade')
        
        
        # Populating the fuzzy variables with terms
        self.GPA.automf(names=self.getEnumNames(fr.FuzzyRules.GPA))
        self.GPAchanging.automf(names=self.getEnumNames(fr.FuzzyRules.GPAchanging))
        self.RangeOfCourses.automf(names=self.getEnumNames(fr.FuzzyRules.RangeOfCourses))
        '''
         There are 12 types of grades combined with the range of courses as following indexes:
            TG_L_Last	[1]
            AG_L_Last	[2]
            LG_L_Last	[3]
            TG_L_All	[4]
            AG_L_All	[5]
            LG_L_All	[6]
            TG_Last		[7]
            AG_Last		[8]
            LG_Last		[9]
            TG_All		[10]
            AG_All		[11]
            LG_All		[12]
        '''
        self.typeOfGrade.automf(names=['1','2','3','4','5','6','7','8','9','10','11','12'])        
        
        self.Rules = []
        self.ruleSystem = None

        return


    def getEnumNames(self, enumObject):
        enumList=[]
        for enumData in enumObject:
            enumList.append(enumData.name)
        return enumList



    def defineRules(self):
        iRule=0
        for eGPAData in fr.FuzzyRules.GPA:
            for eGPAchangingData in fr.FuzzyRules.GPAchanging:            
                for eRangeOfCoursesData in fr.FuzzyRules.RangeOfCourses:
                    iRule+=1
                    #print('{} {} {} {} {} {} {}'.format(iRule, eGPAData.name, eGPAData.value, eGPAchangingData.name, eGPAchangingData.value, eRangeOfCoursesData.name, eRangeOfCoursesData.value))
                    self.Rules.append(self.iniRule(eGPAData.name, eGPAchangingData.name, eRangeOfCoursesData.name, iRule))
        
        self.ruleSystem = ctrl.ControlSystem(rules=self.Rules)
        
        return
        
    
    
    def iniRule(self, eGPADataName, eGPAchangingDataName, eRangeOfCoursesDataName, iRule):
        tGrade = '5'
        '''
            here we need to map all rules from [1 to 96] to the outputs from [1 to 12]
        '''
        return ctrl.Rule(
                antecedent=(
                        self.GPA[eGPADataName] & 
                        self.GPAchanging[eGPAchangingDataName] & 
                        self.RangeOfCourses[eRangeOfCoursesDataName]), 
                consequent=self.typeOfGrade[str(tGrade)], label=str(iRule))
        

    def runSimulation(self):

        sim = ctrl.ControlSystemSimulation(self.ruleSystem, flush_after_run=21 * 21 + 1)
        
        # We can simulate at higher resolution with full accuracy
        upsampled = np.linspace(-2, 2, 21)
        x, y = np.meshgrid(upsampled, upsampled)
        z = np.zeros_like(x)
        
        # Loop through the system 21*21 times to collect the control surface 
        for i in range(21):
            for j in range(21):
                sim.input['GPAchanging'] = x[i, j]
                sim.input['GPA'] = x[i, j]
                sim.input['RangeOfCourses'] = y[i, j]
                sim.compute()
                z[i, j] = sim.output['typeOfGrade']
        
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        
        cset = ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
        cset = ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
        cset = ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)
        
        ax.view_init(30, 200)


def main():

    fC = FuzzyControl()
    fC.defineRules()
    fC.runSimulation()
    
    print ("Process completed.")
    
    
if __name__ == "__main__":
    main()