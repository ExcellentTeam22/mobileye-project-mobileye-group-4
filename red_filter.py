import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

image = np.array(Image.open("image_path"))

# read the image
img = cv2.imread("image_path")

# convert the BGR image to HSV colour space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

# join my masks
mask = mask0 + mask1

# perform bitwise and on the original image arrays using the mask
res = cv2.bitwise_and(img, img, mask=mask)
image_max = ndi.maximum_filter(res, size=20, mode='constant')

plt.figure()
plt.clf()
h = plt.subplot(111)
plt.imshow(image)
plt.figure()
plt.clf()
plt.subplot(111, sharex=h, sharey=h)
plt.imshow(mask, cmap='gray')
plt.figure()
plt.clf()
plt.subplot(111, sharex=h, sharey=h)
plt.imshow(image_max, cmap='gray')