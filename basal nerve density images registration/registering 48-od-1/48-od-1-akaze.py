#Load Dependencies
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL 


#Establish Paths
path = r'Basal nerve density2/Severe/48-od-1/'
outpath = r'Basal nerve density2/Severe/48-od-1/output'

#Establishing Array
registered_images = []
grey_images = []

#Raw to Grey Image
for image in os.listdir(path):
    if ('count' not in image and '.jpg' in image):
        img = cv2.imread(os.path.join(path, image))
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey_images.append(grey_image)


#Establishing Counter
z = 0
y = 33


##AKAZE METHODS

###First Registration###

    #Establishing Image Dimensions 
height, width = 490, 490
    #Orb set
akaze = cv2.AKAZE_create()
kp1 , d1 = akaze.detectAndCompute(grey_images[32], None) 
kp2 , d2 = akaze.detectAndCompute(grey_images[y], None) 

    #Matching
matcher = cv2.BFMatcher() 
matches = matcher.match(d1, d2) 
matches.sort(key = lambda x: x.distance) 
matches = matches[:int(len(matches)*90)] 
no_of_matches = len(matches) 
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
for i in range(len(matches)): 
    p1[i, :] = kp1[matches[i].queryIdx].pt 
    p2[i, :] = kp2[matches[i].trainIdx].pt 
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    #Output Registered Image to Array
transformed_img1 = cv2.warpPerspective(grey_images[y], homography, (width, height)) 
registered_images.append(transformed_img1)

y += 1



###Automation for rest of images###
for grey_image in grey_images:
    if z < 2:
        akaze = cv2.AKAZE_create()
        kp1 , d1 = akaze.detectAndCompute(registered_images[z], None) 
        kp2 , d2 = akaze.detectAndCompute(grey_images[y], None) 

        matcher = cv2.BFMatcher() 
        matches = matcher.match(d1, d2) 
        matches.sort(key = lambda x: x.distance) 
        matches = matches[:int(len(matches)*90)] 
        no_of_matches = len(matches) 


        p1 = np.zeros((no_of_matches, 2)) 
        p2 = np.zeros((no_of_matches, 2)) 
        for i in range(len(matches)): 
            p1[i, :] = kp1[matches[i].queryIdx].pt 
            p2[i, :] = kp2[matches[i].trainIdx].pt 
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
        
        transformed_img = cv2.warpPerspective(grey_images[y], homography, (width, height)) 
        registered_images.append(transformed_img)
        
        y += 1
        z += 1   
  


#Saving Registered Images 

registered_images_arr = np.array(registered_images, dtype = np.uint8)  

num = 0
for output in registered_images_arr:
    cv2.imwrite('Basal nerve density2/Severe/48-od-1/output/registered_akaze' + str(num) + ".jpg", output)
    num += 1