import os
import cv2
import numpy as np
import imutils
import random
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip,Crop,ElasticTransform,RandomBrightnessContrast
random.seed(7)

augRotate = RandomRotate90(p=1)
augDist = GridDistortion(p=1.0)
augFlip = HorizontalFlip(p=1.0)
augFlipV = VerticalFlip(p=1.0)
augElastic = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
augBright = RandomBrightnessContrast(p=1)
class Dataset:

    def __init__(self,path,trainingSetName,testSetName,tup):
        self.path=path
        self.trainingSetName=trainingSetName
        self.testSetName=testSetName
        self.tup=tup
        self.pathList=self.getListOfPaths()


    def getListOfPaths(self):

        rootdirstList=[self.path+"/"+self.trainingSetName,self.path+"/"+self.testSetName]
        listOfPaths=[]
        for rootdir in rootdirstList:
            for subdir, dirs, files in os.walk(rootdir):
                for dir in dirs:
                    filePath=rootdir+os.sep+dir
                    filePathstr=filePath
                    if self.tup=='Segmentation':
                      if filePathstr.split(os.sep)[-1]=='Image':
                        listOfPaths.append(filePath)
                    else:
                      listOfPaths.append(filePath)
    
        return listOfPaths
                 
    def importImages(self):

        trainingListX,testListX,trainingListY,testListY=[],[],[],[]
        counter=1
        for rootdir in self.pathList: 
        
            for subdir,dirs,files in os.walk(rootdir):
                counterin=0
                if counterin==0:
                    splitter=files[0]
                    self.dataType=splitter.split(".")[1]
                for file in files:
                    filestr=file
                    if filestr.split(".")[1] in ['png','jpg','jpeg'] and (")") not in file:
                        filePath=subdir + os.sep + file
                        cropped=cv2.resize(cv2.imread(filePath),(224,224),interpolation=cv2.INTER_CUBIC)
                        strPath=filePath
                        if strPath.split(os.sep)[-2]=='No':
                          counterm=0
                        else:
                          counterm=1
                        if counter<=len(self.pathList)/2:
                        
                            trainingListX.append(cropped)
                            trainingListY.append(counterm)
                            augRotated = augRotate(image=cropped)
                    
                            augElasticed = augElastic(image=cropped)
                            augDistorted = augDist(image=cropped)
                           
                            if counterm:
                              trainingListX.append(augRotated['image'])
                              trainingListY.append(counterm)
                              trainingListX.append(augElasticed['image'])
                              trainingListY.append(counterm)
                              trainingListX.append(augDistorted['image'])
                              trainingListY.append(counterm)

                            else:
                              trainingListX.append(augRotated['image'])
                              trainingListY.append(counterm)
                            
                        else:
                        
                            testListX.append(cropped)
                            testListY.append(counterm)
                    counterin+=1
            counter+=1
        
        trainingArrayX,testArrayX,trainingArrayY,testArrayY=np.array(trainingListX),np.array(testListX),np.array(trainingListY),np.array(testListY)  
        shuffler = np.random.permutation(len(trainingArrayX))
        self.trainingArrayX,self.trainingArrayY=trainingArrayX[shuffler],trainingArrayY[shuffler]
        shufflerTest = np.random.permutation(len(testArrayX))
        self.testArrayX,self.testArrayY=testArrayX[shufflerTest],testArrayY[shufflerTest]

    def importSegmentationClass(self,augmentation):
        trainingListX,testListX,trainingListY,testListY,nameList=[],[],[],[],[]
 
        for i in range(len(self.pathList)):
 
            for subdir,dirs,files in os.walk(self.pathList[i]):
         
                counterin=0
                if counterin==0:
                    splitter=files[0]
                    self.dataType=splitter.split(".")[1]
                for file in files: 
                    filepath=subdir + os.sep + file
                    filestr=file
                    filestr2=file
                    if len(filestr.split("."))==2 and filestr.split(".")[-1] in ['png','pjg','jpeg'] and len(filestr2.split(" "))==1: 
                        splitter=filepath                 
                        splitted=splitter.split(os.sep)
                        splitted=splitted[:len(splitted)-2]
                        splittedstr="/".join([elem for elem in splitted])
                        splittedstr+="/Annotation/"+file
                        cropped=cv2.cvtColor(self.contour_crop_resize(cv2.imread(filepath)),cv2.COLOR_BGR2GRAY)
                        masked=cv2.cvtColor(cv2.imread(splittedstr),cv2.COLOR_BGR2RGB)
                                                
                        if self.pathList[i].split(os.sep)[-2]=='TrainingSet':
                          trainingListX.append(cropped)
                          trainingListY.append(masked)
                          if augmentation:
                            augRotated = augRotate(image=cropped, mask=masked)
                            augDistorted = augDist(image=cropped,mask=masked)
                            
                            trainingListX.append(augRotated['image'])
                            trainingListY.append(augRotated['mask'])
                            trainingListX.append(augDistorted['image'])
                            trainingListY.append(augDistorted['mask'])
                       
                        else:
                          testListX.append(cropped)
                          testListY.append(masked)
                          nameList.append(file)
                    counterin+=1
        trainingArrayX,testArrayX,trainingArrayY,testArrayY,nameArray=np.array(trainingListX),np.array(testListX),np.array(trainingListY),np.array(testListY),np.array(nameList)
        shuffler = np.random.permutation(len(trainingArrayX))
        self.trainingArrayX,self.trainingArrayY=trainingArrayX[shuffler],trainingArrayY[shuffler]
        shufflerTest = np.random.permutation(len(testArrayX))
        self.testArrayX,self.testArrayY=testArrayX[shufflerTest],testArrayY[shufflerTest]
        self.nameArray=nameArray[shufflerTest]


    def contour_crop_resize(self,image):
      '''
      Contour and crop the image (generally used in brain mri images and object detection)
      :param image: cv2 object
      :param n: if n=0, no cropping, else, cropping exists
      '''
      
      grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      gaussianBlurred=cv2.GaussianBlur(grayscale,(5,5),0)
      thresholded=cv2.threshold(gaussianBlurred,45,255,cv2.THRESH_BINARY)[1]
      eroded=cv2.erode(thresholded,None,iterations=2)
      dilated=cv2.dilate(eroded,None,iterations=2)

      contouring=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      contoursGrabbed=imutils.grab_contours(contouring)
      c=max(contoursGrabbed,key=cv2.contourArea)

      extreme_pnts_left=tuple(c[c[:,:,0].argmin()][0])
      extreme_pnts_right=tuple(c[c[:,:,0].argmax()][0])
      extreme_pnts_top=tuple(c[c[:,:,1].argmin()][0])
      extreme_pnts_bot=tuple(c[c[:,:,1].argmax()][0])
    
      new_image=image[extreme_pnts_top[1]:extreme_pnts_bot[1],extreme_pnts_left[0]:extreme_pnts_right[0]]
      returning=np.zeros((256,256,3),dtype=np.uint8)
      returning[extreme_pnts_top[1]:extreme_pnts_bot[1],extreme_pnts_left[0]:extreme_pnts_right[0]]=new_image
      return returning


    def getInfo(self):
        print("""
        Dataset contains: 
            {} Training examples,
            {} Test examples,
            {} Different classes,
        Image properties are:
            Size: {}
            Data type: {}
        """.format(len(self.trainingArrayX),len(self.testArrayX),len(self.pathList)/2,self.trainingArrayX[0].shape,self.dataType))

        





        
        
    
