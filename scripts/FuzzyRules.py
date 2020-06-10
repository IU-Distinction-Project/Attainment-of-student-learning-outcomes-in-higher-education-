"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""

############################################################
from enum import Enum
import numpy as np



class FuzzyRules():

        
    
    ############################################################    
    def __init__(self): return        
    def getFuzzyRulesPredictions(self, Xpred):
        
        YpredFuzzyRules = np.zeros(len(Xpred))
        
        for i in range (len(Xpred)):
            
            S_levelRatio = Xpred[i, 2]
            
            s_GPA = self.getGPAEnum(Xpred[i, 3])            
            GPAChangeRate = self.getGPAChangeRateEnum(Xpred[i, 4])
            
            range_courses = self.getRangeOfcourses(S_levelRatio, s_GPA)            
            indexTypeofGradeAndRangeOfCourses = self.getTypesofGradeFromCube(s_GPA, GPAChangeRate, range_courses)
            
            YpredFuzzyRules[i] = Xpred[i, indexTypeofGradeAndRangeOfCourses]
            
        return YpredFuzzyRules
    
    
    
    
    ############################################################
    def getRangeOfcourses(self, S_levelRatio, s_GPA):        
        '''
        There are four types of ranges:
            ▪ L-Last: All courses (with the same level) studied in the last semester only
            ▪ L-All: All courses (with the same level) previously studied.
            ▪ Last: All courses studied in the last semester only
            ▪ All: All courses previously studied.
        '''        
        if S_levelRatio>.8:
            return self.RangeOfCourses.L_Last
        elif S_levelRatio>.6:
            return self.RangeOfCourses.L_All
        elif S_levelRatio>.4:
            return self.RangeOfCourses.Last
        else:
            return self.RangeOfCourses.All
    
    

    
    ############################################################
    def getTypesofGradeFromCube(self, s_GPA, GPAChangeRate, range_courses):
        
        '''
         There are 12 types of grades combined with the range of courses as following indexes:
            8    X[6]  TG_L_Last
            9    X[7]  AG_L_Last
            10   X[8]  LG_L_Last
            11   X[9]  TG_L_All
            12   X[10] AG_L_All
            13   X[11] LG_L_All
            14   X[12] TG_Last
            15   X[13] AG_Last
            16   X[14] LG_Last
            17   X[15] TG_All
            18   X[16] AG_All
            19   X[17] LG_All
            20   X[18] AA_L_Last
            21   X[19] AA_L_All
            22   X[20] AA_Last
            23   X[21] AA_All
            24   X[22] BA_L_Last
            25   X[23] BA_L_All
            26   X[24] BA_Last
            27   X[25] BA_All
        '''
        # These are 96 rules (4 * 6 * 4)
        if s_GPA is self.GPA.Excellent: return self.withExcellent(GPAChangeRate, range_courses)
        if s_GPA is self.GPA.VeryGood: return self.withVeryGood(GPAChangeRate, range_courses)
        if s_GPA is self.GPA.Good: return self.withGood(GPAChangeRate, range_courses)
        if s_GPA is self.GPA.Pass: return self.withPass(GPAChangeRate, range_courses)
        return None        
        


    ############################################################
    def withExcellent(self, GPAChangeRate, range_courses):
        # Excellent and HI and the four ranges
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.All:return 21
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.Last:return 12        
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_All:return 9
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_Last:return 6        
        # Excellent and MI and the four ranges
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.All:return 21
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.Last:return 20        
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_All:return 19
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_Last:return 6        
        # Excellent and SI and the four ranges
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.All:return 16
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.Last:return 20        
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_All:return 10
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_Last:return 18       
        # Excellent and HD and the four ranges
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.Last:return 14        
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_All:return 11
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_Last:return 8        
        # Excellent and MD and the four ranges
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.Last:return 24        
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_All:return 23
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_Last:return 22        
        # Excellent and SD and the four ranges
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.All:return 16
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.Last:return 24        
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_All:return 10
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_Last:return 22        
        return None
    
    
    
    ############################################################
    def withVeryGood(self, GPAChangeRate, range_courses):
         # VeryGood and HI and the four ranges
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.All:return 21
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.Last:return 12        
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_All:return 9
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_Last:return 6        
        # VeryGood and MI and the four ranges
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.All:return 21
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.Last:return 20       
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_All:return 19
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_Last:return 6        
        # VeryGood and SI and the four ranges
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.All:return 21
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.Last:return 20        
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_All:return 19
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_Last:return 18        
        # VeryGood and HD and the four ranges
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.Last:return 14        
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_All:return 11
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_Last:return 8        
        # VeryGood and MD and the four ranges
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.Last:return 14        
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_All:return 23
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_Last:return 8        
        # VeryGood and SD and the four ranges
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.All:return 16
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.Last:return 24        
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_All:return 10
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_Last:return 22              
        return None
    
    
    
    ############################################################
    def withGood(self, GPAChangeRate, range_courses):
        # Good and HI and the four ranges
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.All:return 21
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.Last:return 12        
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_All:return 9
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_Last:return 7        
        # Good and MI and the four ranges
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.All:return 21
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.Last:return 12        
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_All:return 19
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_Last:return 18        
        # Good and SI and the four ranges
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.All:return 16
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.Last:return 20        
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_All:return 10
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_Last:return 18        
        # Good and HD and the four ranges
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.Last:return 14        
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_All:return 11
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_Last:return 8        
        # Good and MD and the four ranges
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.Last:return 14        
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_All:return 23
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_Last:return 8        
        # Good and SD and the four ranges
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.Last:return 24        
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_All:return 10
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_Last:return 22                
        return None 
    
    
    
    ############################################################
    def withPass(self, GPAChangeRate, range_courses):
        # Pass and HI and the four ranges
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.All:return 15
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.Last:return 12        
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_All:return 9
        if GPAChangeRate is self.GPAchanging.HI and range_courses is self.RangeOfCourses.L_Last:return 6        
        # Pass and MI and the four ranges
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.All:return 16
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.Last:return 12        
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_All:return 10
        if GPAChangeRate is self.GPAchanging.MI and range_courses is self.RangeOfCourses.L_Last:return 6        
        # Pass and SI and the four ranges
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.All:return 16
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.Last:return 20        
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_All:return 10
        if GPAChangeRate is self.GPAchanging.SI and range_courses is self.RangeOfCourses.L_Last:return 18        
        # Pass and HD and the four ranges
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.All:return 17
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.Last:return 14        
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_All:return 11
        if GPAChangeRate is self.GPAchanging.HD and range_courses is self.RangeOfCourses.L_Last:return 8        
        # Pass and MD and the four ranges
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.All:return 17
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.Last:return 14        
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_All:return 11
        if GPAChangeRate is self.GPAchanging.MD and range_courses is self.RangeOfCourses.L_Last:return 22        
        # Pass and SD and the four ranges
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.All:return 25
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.Last:return 24        
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_All:return 23
        if GPAChangeRate is self.GPAchanging.SD and range_courses is self.RangeOfCourses.L_Last:return 22        
        return None
        
        
    
    
    ############################################################    
    def getGPAEnum(self, sGPA):        
        if sGPA>=.90: return self.GPA.Excellent # 4.5
        if sGPA>=.75: return self.GPA.VeryGood  #3.75
        if sGPA>=.55: return self.GPA.Good      #2.75
        return self.GPA.Pass
    
    
    
    ############################################################
    def getGPAChangeRateEnum(self, GPAChangeRate):
        if GPAChangeRate>=1:    return self.GPAchanging.HI
        if GPAChangeRate>=.5:   return self.GPAchanging.MI
        if GPAChangeRate>=0:    return self.GPAchanging.SI
        if GPAChangeRate>=-.2:  return self.GPAchanging.SD
        if GPAChangeRate>=-.5:  return self.GPAchanging.MD       
        return self.GPAchanging.HD
        
        

    
    ############################################################
    class GPA(Enum):
        Excellent = 1
        VeryGood = 2
        Good = 3
        Pass = 4
    
    
    ############################################################
    class GPAchanging(Enum):
        '''
        GPA changing within the last two semesters
            SD: Small decrease
            MD: Medium decrease
            HD: High decrease
            SI: Small increase
            MI: Medium increase
            HI: High increase
        '''
        SD = 1
        MD = 2
        HD = 3
        SI = 4
        MI = 5
        HI = 6        
        
        
    ############################################################
    class RangeOfCourses(Enum):
        L_Last = 1
        L_All = 2
        Last = 3
        All = 4
    




