import os
from skimage import morphology
import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import imutils

class PngGenerator:
  def __init__(self,dicomPath : str,taskType : str):
    self.dicomPath=dicomPath
    self.taskType=taskType
  def insertDicomData(self):
    '''
    Getting the list of paths for the .dcm files inside the path specified in the constructor.
    '''
    self.pathList=[]
    self.nameList=[]
    for subdir,dirs,files in os.walk(self.dicomPath):
      for file in files:
        filestr=file
        filestr2=file
        if len(filestr.split("."))==2 and len(filestr2.split(" "))==1 and filestr.split(".")[1]=='dcm':
          filePath=subdir+os.sep+file
          self.pathList.append(filePath) 
          strFile=file
          fileSplitted=strFile.split(".")
          name=fileSplitted[0]+".png"
          self.nameList.append(name) 

  def getPngData(self,pathToUpperFolder : str,FolderName : str):
    '''
    Creating a folder consists of .png files includes the extracted images from the dicom images to the specified path.
    :param pathToUpperFolder: path to create the upper folder
    :parm FolderName: name of the folder contains the images
    '''
    self.dataList=[]
    path=pathToUpperFolder+"/"+FolderName
    if not os.path.exists(path):
      os.makedirs(path)
    self.pngList=[]
    
    for i in range(len(self.pathList)):

      fileName=self.nameList[i]
      removed=self.remove_noise(self.pathList[i])

      if self.taskType=='Classification':

        plt.imsave(path+os.sep+fileName,removed,cmap='bone')
        plt.imsave(path+os.sep+fileName,self.add_pad(self.contour_crop_resize_two(cv2.imread(path+os.sep+fileName))),cmap='bone')
    
      elif self.taskType=='Segmentation': 

        plt.imsave(path+os.sep+fileName,removed,cmap='bone')
        resized=self.contour_crop_resize(cv2.imread(path+os.sep+fileName))
        plt.imsave(path+os.sep+fileName,resized,cmap='bone')

  def transform_to_hu(self,medical_image,image):
    '''
    Extracts a 'hu' image from dicom file.
    :param medical_image: pydicom object
    :param image: pixel array of the pydicom object
    '''
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

  def window_image(self,image, window_center, window_width):
    '''
    Extracts the image with specified window center and the window width.
    :param image: the pixel array
    :param window_center: Brightness option
    :param window_width: Contrast option
    '''
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    data=(window_image*255/window_image.max()).astype(np.uint8)
    return data

  def remove_noise(self,file_path : str, display=False):
    '''
    A function that uses both transform_to_hu and window_image to extract a image with removing the noises.
    :param file_path: The path to dicom files
    :param display: If true, you can visualize the extraction operation
    '''
    medical_image = pydicom.read_file(file_path)
    image = medical_image.pixel_array
    image12146=image
    if image12146.shape[0]>512:
      image12146=image12146[((image12146.shape[0]-512)//2):((image12146.shape[0]-512)//2)+512,:]
    if image12146.shape[1]>512:
      image12146=image12146[:,((image12146.shape[1]-512)//2):((image12146.shape[1]-512)//2)+512]
    
    image=image12146
    hu_image = self.transform_to_hu(medical_image, image)
    brain_image = self.window_image(hu_image, 40, 40)

    # morphology.dilation creates a segmentation of the image
    # If one pixel is between the origin and the edge of a square of size
    # 5x5, the pixel belongs to the same class
    
    # We can instead use a circule using: morphology.disk(2)
    # In this case the pixel belongs to the same class if it's between the origin
    # and the radius
    
    segmentation = morphology.dilation(brain_image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)
    
    label_count = np.bincount(labels.ravel().astype(np.int))
    # The size of label_count is the number of classes/segmentations found
    
    # We don't use the first class since it's the background
    label_count[0] = 0
    
    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()
    
    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    
    # Since the the pixels in the mask are zero's and one's
    # We can multiple the original image to only keep the brain region
    masked_image = mask * brain_image

    if display:
      plt.figure(figsize=(15, 2.5))
      plt.subplot(141)
      plt.imshow(brain_image)

      plt.title('Original Image')
      plt.axis('off')
      
      plt.subplot(142)

      plt.imshow(mask)
      plt.title('Mask')
      plt.axis('off')

      plt.subplot(143)

      plt.imshow(masked_image)
      plt.title('Final Image')
      plt.axis('off')
    data=masked_image
  
    return data
    
  def add_pad(self,image, new_height=512, new_width=512):
    '''
    A function to add padding to images that doesn't have the required shapes (512*512 required)
    :param image: a cv2 object 
    :param new_height: required height 
    :param new_width: required width
    '''
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape[0],image.shape[1]

    final_image = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)
    
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    
    return final_image

  def contour_crop_resize(self,image):
    '''
    Contour and crop the image (generally used in brain mri images and object segmentation)
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
    returning=np.zeros((512,512,3),dtype=np.uint8)
    returning[extreme_pnts_top[1]:extreme_pnts_bot[1],extreme_pnts_left[0]:extreme_pnts_right[0]]=new_image
    return returning

  def contour_crop_resize_two(self,image):
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
    return new_image
