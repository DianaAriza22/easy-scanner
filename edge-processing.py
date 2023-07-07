import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(image):
    cv2.imshow('image',image)
    c = cv2.waitKey(0)
    if c >= 0 : cv2.destroyAllWindows()
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image',image)

    #key = cv2.waitKey(0)
    #if key == 27:  # If the ESC key is pressed
    #    cv2.destroyAllWindows()


    

# Read the original image
img = cv2.imread('data/rectangle(1).jpg')
show_image(img)

#applying guassian blur
img_blur = cv2.GaussianBlur(img,(5, 5),cv2.BORDER_DEFAULT)

show_image(img_blur)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

show_image(edges)



