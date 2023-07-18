import math
from PIL import Image
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import io, color, filters
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

def show_image(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',image)
    c = cv2.waitKey(0)
    if c >= 0 : cv2.destroyAllWindows()
    

# Function to rotate an image clockwise
def rotate_clockwise(image, degrees):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -degrees, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def find_axis_crossings(point, slope):
    x, y = point

    # Equation of the line in point-slope form: y - y1 = m(x - x1)
    # where (x1, y1) is the given point and m is the slope
    # Rearrange the equation to slope-intercept form: y = mx + (y1 - mx1)
    intercept = y - slope * x

    # Crossing with the x-axis (y = 0)
    x_axis_crossing = -intercept / slope

    # Crossing with the y-axis (x = 0)
    y_axis_crossing = intercept

    return x_axis_crossing, y_axis_crossing

    
# Read the original image
img = cv2.imread('rotated_image.jpg')
im = Image.open('rotated_image.jpg')


#applying guassian blur
img_blur = cv2.GaussianBlur(img,(5, 5),cv2.BORDER_DEFAULT)
#show_image(img_blur)

#applying Canny Filtering
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
show_image(edges)

# Find contours
contours, hierarchy  = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#Show contours
img_blur2 = cv2.GaussianBlur(img,(41, 41),cv2.BORDER_DEFAULT) # High blur to highlight the contour
img_res = cv2.drawContours(img_blur2, contours, -1, (0,0,0), 2) #(0,255,75)
#show_image(img_res)

# RGB to B/W conversion
img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
#show_image(img_res)

edges2 = cv2.Canny(image=img_res, threshold1=100, threshold2=200)
show_image(edges2)

print(f"len {len(img_res)}")
print(f"type{type(img_res)}")
print(f"dim{img_res.ndim}")

# Hough transform 
tested_angles_h = np.linspace(- np.pi /2, np.pi / 2, 200)
h_h, theta_h, d_h = hough_line(edges2, theta=tested_angles_h)

# Extract prominent lines from the Hough accumulator using a threshold
_, angles, distances = hough_line_peaks(h_h, theta_h, d_h, threshold=100)

rotated_image = img
rotated_image2 = img

#show_image(rotated_image)
#show_image(rotated_image2)
#show_image(edges2)
#print(f"dim_edges2: {edges2.ndim}")

# Plot the input image and the detected horizontal lines
fig, ax = plt.subplots(1, 1, figsize=(10, 5))



ax.imshow(edges2, cmap=cm.gray)
ax.set_ylim((edges2.shape[0], 0))
ax.set_axis_off()


for angle, dist in zip(angles, distances):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax.axline((x0, y0), slope=np.tan(angle + np.pi/2))

#print(f"x0,y0: {x0},{y0}")

plt.tight_layout()
plt.savefig("detected_lines.png", bbox_inches="tight",pad_inches=0, dpi=100)
plt.show()


#print(np.rad2deg(angles))
#print((distances))
#print(len(angles))
#(x0, y0) = distances * np.array([np.cos(angles), np.sin(angles)])
#slopes = np.tan(angles + np.pi/2)


intersections = []
for i in range(len(angles)):
    for j in range(i+1, len(angles)):
        angle1 = angles[i]
        angle2 = angles[j]
        dist1 = distances[i]
        dist2 = distances[j]

        # Calculate the intersection point
        x = (dist2*np.sin(angle1) - dist1*np.sin(angle2)) / np.sin(angle1 - angle2)
        y = (dist1 - x*np.cos(angle1)) / np.sin(angle1)
        intersections.append((x, y))

#Print the intersection points
for intersection in intersections:
    print(f"Intersection: {intersection}")
print(len(intersections))
#print(intersections[:,0])
show_image(img)

x, y = zip(*intersections)

#print(intersections[1])

#x = intersections[:,0]
#y = intersections[:,1]

intersections_index = []

for i in range(len(x)-1):
    if x[i] > 0 and math.isinf(x[i]) == False:
        intersections_index.append(i)

for i in range(len(y)-1):
    if y[i] > 0 and math.isinf(y[i]) == False:
        intersections_index.append(i)

print(f"Intersections_index:{intersections_index}")
intersections_index2 = [*set(intersections_index)]
print(f"Intersections_index2:{intersections_index2}")
new_intersections = []

for i in range(0, len(intersections_index2)):
    new_intersections.append(intersections[intersections_index2[i]])

print(new_intersections)

x, y = zip(*new_intersections)

# Setting the points for cropped image
left = int(min(x))
top = int(max(y))
right = int(max(x))
bottom = int(min(y))

print(f"left{left} top{top} right{right} bottom{bottom}")

im1 = im.crop((left, bottom, right,top))

#print(im1.size)

plt.imshow(im1)
plt.show()

# Shows the image in image viewer
#im1.show()
#show_image(im1)
#crop_img = img[left:left + right, bottom:top]
#show_image(crop_img)

#x_axis_crossing, y_axis_crossing = find_axis_crossings((x0, y0), slopes)
#print(x_axis_crossing)
#tuple(x_axis_crossing)
#tuple(y_axis_crossing)

#x = (int(x_axis_crossing[1]-1000000), 0)
#y = (0, 500)
#print(x)
#x = (50,500)
#y = (555,-555)

#print(edges2.shape)
#new = cv2.line(edges2, x, y, (0, 255, 0), 8)

#show_image(new)

#line = plt.axline((3, 8), slope=0.5, linewidth=4, color='r')
#print(line.get_data())
#plt.plot(line.get_data(),)
#plt.xlim(-10, 10)
#plt.show()