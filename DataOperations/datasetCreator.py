import os
import cv2
import shutil
import numpy as np
from PIL import Image
class DataSetCreator:

    def __init__(self,pathToCreate,datasetName,classNamesList,trainTestRatio):

        self.trainTestRatio=trainTestRatio
        self.datasetName=datasetName
        self.pathToCreate=pathToCreate
        self.classNamesList=classNamesList
        self.createFolderStructure()

    def createFolderStructure(self):

        try:

            pathway=self.pathToCreate+os.sep+self.datasetName

            if not os.path.exists(pathway):
                os.makedirs(pathway)
                self.trainingPath=pathway+os.sep+"TrainingSet"
                os.makedirs(self.trainingPath)
                self.testPath=pathway+os.sep+"TestSet"
                os.makedirs(self.testPath)
                self.trainingPathList=list()
                self.testPathList=list()
                for i in range(len(self.classNamesList)):
                    trainingPath=pathway+os.sep+"TrainingSet"+os.sep+self.classNamesList[i]
                    testPath=pathway+os.sep+"TestSet"+os.sep+self.classNamesList[i]
                    self.trainingPathList.append(trainingPath)
                    self.testPathList.append(testPath)
                    os.makedirs(trainingPath)
                    os.makedirs(testPath)
            else:
                self.trainingPath=pathway+os.sep+"TrainingSet"
                self.testPath=pathway+os.sep+"TestSet"
                self.trainingPathList=list()
                self.testPathList=list()
                for i in range(len(self.classNamesList)):
                    trainingPath=pathway+os.sep+"TrainingSet"+os.sep+self.classNamesList[i]
                    testPath=pathway+os.sep+"TestSet"+os.sep+self.classNamesList[i]
                    self.trainingPathList.append(trainingPath)
                    self.testPathList.append(testPath)

        except:
            raise Exception("Error! Please use valid names and check your variable types.")
        

    def insertImage(self,imagesPath,className):

        number=self.classNamesList.count(className)
 
        if number==0:
            raise Exception("This class doesn't exist. If you want to add this class to your dataset you can use addClass function.")
        else:
            print("here")
            for subdir,dirs,files in os.walk(imagesPath):
                print(dirs)
                counterin=0
                for file in files:
                    filestr=file
                    filestr2=file
                    print("here")
                    if len(filestr.split("."))==2 and len(filestr2.split(" "))==1 and filestr.split(".")[1] in ['png','jpg','jpeg']:
                
                        filePath=subdir + "/" + file
                        if counterin<len(files)*self.trainTestRatio: 
                            cv2.imwrite(self.trainingPath+os.sep+className+os.sep+file,cv2.imread(filePath))
                        else: 
                            cv2.imwrite(self.testPath+os.sep+className+os.sep+file,cv2.imread(filePath))
                    counterin+=1
                    if len(filestr.split("."))==2 and len(filestr2.split(" "))==1 and filestr.split(".")[1]=='dcm':
                        print("burada")
                        filePath=subdir+os.sep+file
                        if counterin<len(files)*self.trainTestRatio:                 
                            if os.path.exists(filePath):

                                # Set the directory path where the file will be moved

                                destination_path = self.trainingPath+os.sep+className

                                # Move the file to the new location

                                new_location = shutil.copy(filePath, destination_path)
                            else:
                                print("file does not exist")
                        else:
                            if os.path.exists(filePath):

                                # Set the directory path where the file will be moved

                                destination_path = self.testPath+os.sep+className

                                # Move the file to the new location

                                new_location = shutil.copy(filePath, destination_path)
                            
                            else:
                                print("file does not exist")

    def addClass(self,className):
        trainingPath=self.trainingPath+os.sep+className
        testPath=self.trainingPath+os.sep+className
        if not os.path.exists(trainingPath):
            os.makedirs(trainingPath)
            self.trainingPathList.append(className)
        if not os.path.exists(testPath):
            os.makedirs(testPath)
            self.testPathList.append(testPath)


            
        


       



