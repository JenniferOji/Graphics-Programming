import cv2
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import copy
#CNN EDGE DETECTION

# img = cv2.imread('ATU.jpg',)
img = cv2.imread('ATU1.jpg',)

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# to control the amount of rows and columns 
nrows = 2
ncols = 3

#the orignal image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# creating a copy of the orignal image so it doesnt affect the orignal 
imgHarris = copy.deepcopy(img)
# gray scale image
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Gray Image'), plt.xticks([]), plt.yticks([])

# harris corner detection 
# 2 = block size 3 = apeture size
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)

threshold = 0.1; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(0, 0, 255),-1)

plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Harris Corner Detection'), plt.xticks([]), plt.yticks([])

plt.show()
