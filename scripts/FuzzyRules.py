"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""

############################################################
from Student import Student
from enum import Enum
import numpy as np



class FuzzyModel():

        
    
    ############################################################    
    def __init__(self, DGapRatioForAT_AL=1.5):         
        
        self.dGapRatioForAT_AL = DGapRatioForAT_AL
        self.studentsList = {}
        self.cube = self.loadCubeRules()
        

    
    
    ############################################################
    def train(self, 
              StudentID,
              StudentLevel,
              CourseID, 
              CourseLevel, 
              CourseCategory, 
              ActualGrade):
        
        # get student details from the list 
        student = self.studentsList.get(StudentID)
        
        if student is None:
            student = Student(StudentID, StudentLevel)
            self.studentsList[StudentID]= student
        
        # then adding a course for training and return the number of added courses already ...        
        student.addCourseAndMark(
                CourseID, 
                CourseLevel,
                CourseCategory,
                ActualGrade)
    
    
    
    ############################################################
    def getFuzzyPredictions(self, Xpred):
   
        YpredFuzzy = np.zeros(len(Xpred))
        
        for i in range (len(Xpred)):

            StudentNextLevel = Xpred[i,2]
            
            ###### No prediction for the entry students
            if StudentNextLevel<=1:                
                YpredFuzzy[i] = .0
                continue
            
            
            # get student object
            StudentID = Xpred[i,0]
            student = self.studentsList.get(StudentID)
            
            if student is None:
                raise ValueError("No training instances for this student! ({})".format(StudentID))        
            
            
            CourseID = Xpred[i,1]
            CourseLevel = Xpred[i,3]
            CourseCategory = Xpred[i,4]
            
            
            # Details from the (Student Object)     
            selectedCourses = self.getTypeOfSelectedCourses(student)
            
            
            # Grade type from the cube
            GradeType= self.getGradeType(
                    student, 
                    selectedCourses)
            
            
            # Assigning predicted grade value..
            YpredFuzzy[i] = student.predict(
                            CourseID, 
                            CourseLevel,                             
                            CourseCategory, 
                            selectedCourses,
                            GradeType, 
                            self.dGapRatioForAT_AL)
            
        return YpredFuzzy
    
    
    
    ############################################################    
    def getTypeOfSelectedCourses(self, student):
        

        # default return
        selectedCourses = self.RangeOfCourses.Last.value
		'''
            There are four types of ranges:
            ▪ C-Last: All courses (within the same category) studied in the last semester only
            ▪ C-All: All courses (within the same category) previously studied.
            ▪ Last: All courses studied in the last semester only
            ▪ All: All courses previously studied.            
        '''        
        # this is for C-Last and C-All
        if student.iCurrentSemester >= 5:
            
            if student.getCurrentGPA() > 4.0:
                selectedCourses = self.RangeOfCourses.C_All.value
            
            #Note  performance of [C-Last] is not good at all, so we used the default .. (Last.value)
        
         # this is for Last and All            
        elif student.iCurrentSemester < 3 or student.getCurrentGPA()<3:            
            selectedCourses = self.RangeOfCourses.All.value

        return selectedCourses
    
    
    
    
    ############################################################
    def getGradeType(self, 
                     student, 
                     selectedCourses):
        
        # Cube shpe (4,4,6) - dimensions [GPA, selectedCourses, gpaChange]
        gpa = student.getGPAEnum().value
        gpaChange = student.getGPAChangeRateEnum().value
        return student.GradeTypes(
                self.cube[
                        gpa, 
                        selectedCourses, 
                        gpaChange])
    

    
    ############################################################
    def loadCubeRules(self):
        
        g = Student(None, None)        
        AT = g.GradeTypes.AT.value
        TG = g.GradeTypes.TG.value
        AA = g.GradeTypes.AA.value
        AG = g.GradeTypes.AG.value
        BA = g.GradeTypes.BA.value
        LG = g.GradeTypes.LG.value
        BL = g.GradeTypes.BL.value        
        cubeRules = []
        
        # Pass Dim = 0
        ############################################
        cubeRules.append([
                [LG, LG, BL, LG, LG, BA],
                [LG, BL, BL, LG, BA, AG],
                [AA, AA, AA, BA, AA, AA], 
                [BA, AA, AA, AT, AA, AA]])

       
        # Good Dim = 1
        ############################################
        cubeRules.append([
                [BA, LG, LG, AG, AG, AA], 
                [AG, LG, BL, AG, AA, TG], 
                [AA, AA, LG, AG, LG, AA], 
                [AA, AA, AG, AA, AA, AG]])

       
        # Very Good Dim = 2
        ############################################
        cubeRules.append([
                [TG, AG, BA, AA, TG, TG], 
                [AG, BA, LG, TG, TG, AT], 
                [AA, AT, BA, AG, AG, TG], 
                [AG, AT, AG, AG, BA, AG]])

				
        # Excellent Dim = 3
        ############################################
        cubeRules.append([
                [TG, AG, BA, TG, TG, TG], 
                [AA, AG, LG, TG, TG, AT], 
                [AT, AA, AT, AT, AA, AT], 
                [AT, AA, AT, AT, AA, AT]])        


        return np.array(cubeRules)




    ############################################################
    class RangeOfCourses(Enum):
        C_Last = 0
        C_All = 1
        Last = 2
        All = 3
        
    




