import cv2
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import copy
#CNN EDGE DETECTION

# img = cv2.imread('ATU.jpg',)
img = cv2.imread('ATU1.jpg',)

# creating a copy of the orignal image so it doesnt affect the orignal 
imgHarris = copy.deepcopy(img)

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# creating a copy of the gray image so it doesnt affect the orignal 
imgShiTomasi = copy.deepcopy(gray_image)
orbImg = copy.deepcopy(gray_image)

# to control the amount of rows and columns 
nrows = 2
ncols = 3

#the orignal image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

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

# shi tomasi algorithm
# how corners are detected 
# 0.01 = quality level - 10 = min distance - 30 = max corners 
corners = cv2.goodFeaturesToTrack(gray_image,150,0.01,10)
corners = np.int0(corners) #convert corners values to integer

for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(255, 0, 0),-1)

plt.subplot(nrows, ncols,4),plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Shi Tomasi algorithm'), plt.xticks([]), plt.yticks([])

# orb detection 
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(orbImg,None)

# compute the descriptors with ORB
kp, des = orb.compute(orbImg, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(orbImg, kp, None, color=(0,255,0), flags=0)

plt.subplot(nrows, ncols,5),plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('ORB image'), plt.xticks([]), plt.yticks([])

plt.show()
