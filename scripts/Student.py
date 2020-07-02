"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""
from enum import Enum
import numpy as np

############################################################       
class Course():
    
    def __init__(self, CourseLevel, CourseCategory, DMark):
       self.courseLevel = CourseLevel       
       self.courseCategory = CourseCategory
       self.dMark = DMark 
           
    def getLevel(self):
       return self.courseLevel

    def getMark(self):
       return self.dMark
    
    # Core, elective, ... [1, 2, 3]
    def getCategory(self):
       return self.courseCategory




############################################################
class Student():
        
    #########################
    def __init__(self, StdID, StudentLevel): 
        self.stdID = StdID
        self.listCourses = {}
        self.iCurrentSemester = StudentLevel;
        self.GPAChangeRate =.0
        self.setGPA = []
        return  
        
 
    
    ##################################################  
    def addCourseAndMark(self,
                         courseID, 
                         courseLevel, 
                         courseCategory,
                         dMark):

        if self.getCurretnLevelBasedOnAddedCourses()>=courseLevel:
            raise ValueError("Attempting to add a old course after computing the GPA")        

        if self.getCurretnLevelBasedOnAddedCourses() + 2 <courseLevel:
            raise ValueError("Attempting to add a new course from the upcoming levels (+2) ! Here, the prediction works in an adaptive process")        
                                        
        if self.listCourses.get(courseID):
            raise ValueError("Attempting to add an existing course")        
            
        
        # adding a new studied course ..
        self.listCourses[courseID] = Course(
                courseLevel, 
                courseCategory, 
                dMark)


        # update GPA when the next level is reached
        if self.getCurretnLevelBasedOnAddedCourses() + 2 == courseLevel:
            self.updateGPA(courseLevel)
            
            
        return len(self.listCourses)


        
    # update GPA
    ##################################################
    def updateGPA(self, nextLevel):
                
        if self.getCurretnLevelBasedOnAddedCourses()+2 != nextLevel:
            return
        
        listMarksForAllPastSemesters =[]
        for course in self.listCourses.values():
            if nextLevel > course.getLevel():
                 listMarksForAllPastSemesters.append(course.getMark())
        
        self.setGPA.append(
                np.mean(listMarksForAllPastSemesters) * 5.0)

        if (len(self.setGPA)>1):
            self.GPAChangeRate = self.setGPA[-1] - self.setGPA[-2]
         
    
    ##################################################
    def getCurrentGPA(self):
        if (len(self.setGPA)>0):
            return self.setGPA[-1]
        return 0.0
    
    
    def getCurretnLevelBasedOnAddedCourses(self):
        return len(self.setGPA)
    
    
    
    
    
    
    ##################################################
    def predict(self,
                courseID, 
                courseLevel, 
                CourseCategory,
                selectedCourses,
                GradeType, 
                dGapRatioForAT_AL=1.5,
                actualMark=None):
        
        if self.iCurrentSemester >= courseLevel:
           raise ValueError("Error: Predictions are allowed only for the upcoming semesters only .. ")


        self.updateGPA(courseLevel)
        
        
        # get the list of courses ...
        listMarksSelectedCourses = self.getMarksFromSelectedCourses(
                selectedCourses, 
                CourseCategory)
        
        
        # compute the predicted grade ..
        dPredGrade = self.getGrade (
                GradeType, 
                listMarksSelectedCourses, 
                dGapRatioForAT_AL)
        
        # compute best mark here .. return must be ["BestGradeType, BestPredMark, isGoodPred"]
        strTracingOfBestMark = self.getBestMappedGrade(
                listMarksSelectedCourses,
                dGapRatioForAT_AL,
                dPredGrade, 
                GradeType, 
                actualMark) 
        
        self.addCourseAndMark(
                courseID, 
                courseLevel, 
                CourseCategory,
                dPredGrade)
        if strTracingOfBestMark is None:
            return dPredGrade
        
        return dPredGrade, strTracingOfBestMark
    


    ##################################################
    def getMarksFromSelectedCourses(
            self, 
            selectedCourses,
            CourseCategory):        
        '''
            C_Last = 0
            C_All = 1
            Last = 2
            All = 3
        '''        
        if selectedCourses<0 or selectedCourses>3:
            raise ValueError("Error: no class available for {}".format(selectedCourses)) 
            
            
        listMarksSelectedCourses = []
        
        # Within just the last semester ..   (C-Last or Last)             
        if selectedCourses == 0 or selectedCourses == 2:
            for course in self.listCourses.values():
                if self.getCurretnLevelBasedOnAddedCourses()==course.getLevel():
                    if selectedCourses == 2 or (CourseCategory==course.getCategory()):
                        listMarksSelectedCourses.append(course.getMark())  
        
        # Within all the previous semester .. (All or C-All)                
        else:
            for course in self.listCourses.values():
                if self.getCurretnLevelBasedOnAddedCourses()>=course.getLevel():
                    if selectedCourses == 3 or (CourseCategory==course.getCategory()):
                        listMarksSelectedCourses.append(course.getMark())  
        
        if len(listMarksSelectedCourses)<1:
            return self.getMarksFromSelectedCourses(3, CourseCategory)
        
        return listMarksSelectedCourses
    
    
    
    
    #########################################################
    def getGrade(self, GradeType, listMarks, dGapRatio):
        
        # Default value
        dPredMark = np.mean(listMarks)
        
        if GradeType is self.GradeTypes.AT:
            dGap = np.max(listMarks) - dPredMark
            dPredMark= dPredMark + (dGap * dGapRatio)
        elif GradeType is self.GradeTypes.TG:
            dPredMark= np.max(listMarks)
        elif GradeType is self.GradeTypes.AA:
            dPredMark= (dPredMark + np.max(listMarks))/2            
        elif GradeType is self.GradeTypes.BA:
            dPredMark= (dPredMark + np.min(listMarks))/2
        elif GradeType is self.GradeTypes.LG:
            dPredMark= np.min(listMarks)       
        elif GradeType is self.GradeTypes.BL:
            dGap= np.min(listMarks) - dPredMark
            dPredMark= dPredMark + (dGap * dGapRatio)          
        
        # grade correction ..
        if dPredMark>1:
            dPredMark =1.0
        elif dPredMark<0:
            dPredMark=.0
        
        return self.roundIt(dPredMark)
        
        
    ############################################################    
    def getGPAEnum(self):        
        if self.getCurrentGPA()>=4.5: return self.eGPA.Excellent # 4.5
        if self.getCurrentGPA()>=3.75: return self.eGPA.VeryGood  #3.75
        if self.getCurrentGPA()>=2.75: return self.eGPA.Good      #2.75
        return self.eGPA.Pass
    
    
    
    ############################################################
    def getGPAChangeRateEnum(self):
        if self.GPAChangeRate>=1:    return self.GPAchanging.HI
        if self.GPAChangeRate>=.5:   return self.GPAchanging.MI
        if self.GPAChangeRate>=0:    return self.GPAchanging.SI
        if self.GPAChangeRate>=-.5:  return self.GPAchanging.SD
        if self.GPAChangeRate>=-.1:  return self.GPAchanging.MD       
        return self.GPAchanging.HD    



    ############################################################
    def Print(self, iLevel=None):
        
        print ("iCurrentSemester: {}\t GPA: {}\t GPAChangeRate: {}".format(
                self.iCurrentSemester, 
                self.getCurrentGPA(), 
                self.GPAChangeRate))
        print ("Sef of all GPAs: {}".format(self.setGPA))
        
        for cID, cV in self.listCourses.items():
            if iLevel is not None and iLevel !=cV.getLevel():
                continue
            print("\t CourseID: {}\t CourseLevel: {}\t CourseCategory: {}\t DMark: {}".format(
                    cID, 
                    cV.getLevel(), 
                    cV.getCategory(), 
                    cV.getMark()))
        return
    
    
    
    ############################################################
    def roundIt(self, xValue):
        return np.round(xValue, 2)
    
    
    
    
    def getBestMappedGrade(self, 
                              listMarksSelectedCourses,
                              dGapRatio,
                              dPredGradeValue, 
                              predGradeType, 
                              actualMark):
                
        if actualMark is None:
            return None
        
        
        minDiff= 101.0
        BestGradeType = None        
        allPossiblePred = {}
        
        for gType in self.GradeTypes:
            allPossiblePred[gType.name] = self.getGrade(
                    gType, 
                    listMarksSelectedCourses, 
                    dGapRatio) 
            
            # find the best pred
            if abs(allPossiblePred[gType.name] - actualMark) < minDiff:
                minDiff = abs(allPossiblePred[gType.name] - actualMark)
                BestGradeType = gType
 
        return "{},{},{}".format(
                BestGradeType.name, 
                allPossiblePred[BestGradeType.name], 
                (BestGradeType.value == predGradeType.value))
        
       
       
       
       
           
    
    
    
    ############################################################
    class GradeTypes(Enum):
        AT = 3
        TG = 2
        AA = 1
        AG = 0
        BA = -1
        LG = -2
        BL = -3


    ############################################################
    class eGPA(Enum):
        Excellent = 3
        VeryGood = 2
        Good = 1
        Pass = 0
    
    
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
        SD = 0
        MD = 1
        HD = 2
        SI = 3
        MI = 4
        HI = 5 









def main():

    s = Student(303, 4)
        
    s.addCourseAndMark("SEMI0012995",1, 1, .78)
    s.addCourseAndMark("SEMI0015183",1, 2, .70)
    s.addCourseAndMark("SEMI0018371",1, 3, .95)
    s.addCourseAndMark("SEMI0015910",1, 1, .69)
    s.addCourseAndMark("SEMI0016208",1, 2, .79)
    s.addCourseAndMark("SEMI0017431",1, 3, .85)
    s.addCourseAndMark("SEMI0014130",1, 1, .72)       
    #s.Print(1)
    
    s.addCourseAndMark("SEMI0024747",2, 1,.72)
    s.addCourseAndMark("SEMI0025909",2, 1,.71)
    s.addCourseAndMark("SEMI0022443",2, 1,.68)
    s.addCourseAndMark("SEMI0025077",2, 1,.73)
    s.addCourseAndMark("SEMI0029604",2, 1,.67)
    s.addCourseAndMark("SEMI0029061",2, 1,.73)
    s.addCourseAndMark("SEMI0022256",2, 1,.83)    
    #s.Print(2)  
       
    s.addCourseAndMark("SEMI0037924",3, 2, .66)
    s.addCourseAndMark("SEMI0034580",3, 2, .59)
    s.addCourseAndMark("SEMI0033576",3, 2, .82)
    s.addCourseAndMark("SEMI0031818",3, 2, .60)
    s.addCourseAndMark("SEMI0039951",3, 2, .67)
    s.addCourseAndMark("SEMI0039037",3, 2, .73)
    s.addCourseAndMark("SEMI0035955",3, 2, .75)
    #s.Print(3)  
     
    s.addCourseAndMark("SEMI0044637",4, 2, .74)
    s.addCourseAndMark("SEMI0048491",4, 2, .62)
    s.addCourseAndMark("SEMI0049197",4, 1, .63)
    s.addCourseAndMark("SEMI0044506",4, 2, .84)
    s.addCourseAndMark("SEMI0044518",4, 2, .80)
    s.addCourseAndMark("SEMI0044275",4, 1, .76)
    s.Print(4)
    '''
    s.predict("SEMI0058624",5, 2, s.GradeTypes.AG, 2, 1.5)
    s.predict("SEMI0058693",5, 2, s.GradeTypes.TG, 2, 1.5)
    s.predict("SEMI0056942",5, 2, s.GradeTypes.TG, 2, 1.5)
    s.predict("SEMI0055343",5, 1, s.GradeTypes.TG, 2, 1.5)
    s.predict("SEMI0052593",5, 2, s.GradeTypes.TG, 2, 1.5)
    s.predict("SEMI0055015",5, 1, s.GradeTypes.TG, 2, 1.5)
    s.predict("SEMI0052434",5, 2, s.GradeTypes.TG, 2, 1.5)
    s.Print(5)
   
    s.predict("SEMI0068923",6, 3, 0, s.GradeTypes.AG, 1.5)
    s.predict("SEMI0067259",6, 2, s.GradeTypes.AG, 2, 1.5)
    s.predict("SEMI0065623",6, 2, s.GradeTypes.AG, 2, 1.5)
    s.predict("SEMI0064181",6, 2, s.GradeTypes.AG, 2, 1.5)
    s.predict("SEMI0064427",6, 2, s.GradeTypes.AG, 2, 1.5)
    s.predict("SEMI0069632",6, 2, s.GradeTypes.AG, 2, 1.5)
    s.predict("SEMI0069737",6, 2, s.GradeTypes.AG, 2, 1.5)
    s.Print(6)
    
    def predict(self,
                courseID, 
                courseLevel, 
                CourseCategory,
                selectedCourses,
                GradeType, 
                dGapRatioForAT_AL=1.5,
                actualMark=None):
        
    '''
    courseID = "SEMI0058624"
    courseLevel = 5 
    CourseCategory = 2
    selectedCourses = 3
    GradeType = s.GradeTypes(0)
    dGapRatioForAT_AL=1.2
    actualMark=0.65
                
      
    _,traceStr = s.predict(
            courseID,
            courseLevel, 
            CourseCategory, 
            selectedCourses, 
            GradeType, 
            dGapRatioForAT_AL,
            actualMark)
    
    print (traceStr)
    
    #s.Print()
    '''
    s.addCourseAndMark("SEMI0058624",5, 2, .85)
    s.addCourseAndMark("SEMI0058693",5, 2, .67)
    s.addCourseAndMark("SEMI0056942",5, 2, .57)
    s.addCourseAndMark("SEMI0055343",5, 2, .66)
    s.addCourseAndMark("SEMI0052593",5, 2, .80)
    s.addCourseAndMark("SEMI0055015",5, 2, .81)
    s.addCourseAndMark("SEMI0052434",5, 2, .66)
    s.Print(5)
    
    s.addCourseAndMark("SEMI0068923",6, 2, .74)
    s.addCourseAndMark("SEMI0067259",6, 2, .76)
    s.addCourseAndMark("SEMI0065623",6, 2, .72)
    s.addCourseAndMark("SEMI0064181",6, 2, .73)
    s.addCourseAndMark("SEMI0064427",6, 2, .75)
    s.addCourseAndMark("SEMI0069632",6, 2, .69)
    s.addCourseAndMark("SEMI0069737",6, 2, .85)
    s.Print(6)
    
    s.addCourseAndMark("SEMI0076821",7, 2, .63)
    s.addCourseAndMark("SEMI0071868",7, 2, .87)
    s.addCourseAndMark("SEMI0077220",7, 2, .72)
    s.addCourseAndMark("SEMI0078668",7, 2, .72)
    s.addCourseAndMark("SEMI0073367",7, 2, .77)
    s.addCourseAndMark("SEMI0079276",7, 2, .45)
    s.addCourseAndMark("SEMI0078510",7, 2, .65)
    s.Print(7)
    
    s.addCourseAndMark("SEMI0089747",8, 2, .61)
    s.addCourseAndMark("SEMI0081989",8, 2, .75)
    s.addCourseAndMark("SEMI0082598",8, 2, .52)
    s.addCourseAndMark("SEMI0088030",8, 2, .77)
    s.addCourseAndMark("SEMI0081794",8, 2, .52)
    s.addCourseAndMark("SEMI0086600",8, 2, .68)
    s.addCourseAndMark("SEMI0083259",8, 2, .54)
    s.Print(8)
    #s.Print()
    '''
    print ("Testing completed.")
    
    
if __name__ == "__main__":
    main()