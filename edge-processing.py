import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image)
    c = cv2.waitKey(0)
    if c >= 0 : cv2.destroyAllWindows()
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image',image)

    #key = cv2.waitKey(0)
    #if key == 27:  # If the ESC key is pressed
    #    cv2.destroyAllWindows()


    
# Read the original image
img = cv2.imread('data/data(11).jpg')
#show_image(img)

#applying guassian blur
img_blur = cv2.GaussianBlur(img,(5, 5),cv2.BORDER_DEFAULT)
#show_image(img_blur)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
show_image(edges)

contours, hierarchy  = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


perimeter = []
for i in range(0,len(contours)):
    perimeter.append(cv2.arcLength(contours[i], False))
    
print(max(perimeter))
idx_max_perimeter = perimeter.index(max(perimeter))
print(idx_max_perimeter)

img_res = cv2.drawContours(img, contours, idx_max_perimeter, (0,255,75), 2)
show_image(img_res)

#area0 = cv2.contourArea(contours[1])
#perimeter = cv2.arcLength(contours[1], False)
#print(area0)
#print(perimeter)
#print(hierarchy) #[Next, Previous, First_Child, Parent]
#print(contours[1]) 


