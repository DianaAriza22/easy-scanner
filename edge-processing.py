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
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


    
# Read the original image
img = cv2.imread('data/data(42).jpg')

#applying guassian blur
img_blur = cv2.GaussianBlur(img,(9, 9),cv2.BORDER_DEFAULT)
#show_image(img_blur)

#applying Canny Filtering
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
show_image(edges)

# Find contours
contours, hierarchy  = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



# Find longest contour
perimeter = []
for i in range(0,len(contours)):
    perimeter.append(cv2.arcLength(contours[i], False))

sorted_indices = np.argsort(perimeter)[::-1]

largest_contours = []
number_of_contours = 10
for i in range(0,number_of_contours):
    largest_contours.append(contours[sorted_indices[i]])

    
#Show longest contour
img_blur2 = cv2.GaussianBlur(img,(41, 41),cv2.BORDER_DEFAULT) # High blur to highlight the contour
img_res = cv2.drawContours(img_blur2, largest_contours, -1, (0,0,0), 2) #(0,255,75)
show_image(img_res)

# RGB to B/W conversion
img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
show_image(img_res)

edges2 = cv2.Canny(image=img_res, threshold1=100, threshold2=200)
show_image(edges2)

print(f"len {len(img_res)}")
print(f"type{type(img_res)}")
print(f"dim{img_res.ndim}")

# Hough transform 
tested_angles_h = np.linspace(- np.pi /2, np.pi / 2, 360, endpoint=False)
h_h, theta_h, d_h = hough_line(edges2, theta=tested_angles_h)

# Extract prominent lines from the Hough accumulator using a threshold
_, angles, distances = hough_line_peaks(h_h, theta_h, d_h, threshold=100)
print(np.rad2deg(angles))

# Perform K-means clustering to find the main angles
if len(angles) > 1:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(angles.reshape(-1, 1))
    main_angles = kmeans.cluster_centers_.flatten()
else:
    main_angles = angles
print('testing main  ',np.rad2deg(main_angles))
main_angles=np.rad2deg(main_angles)


#///////////////////////////////////////
# Calculate the means of angles close to the main angles
#means = []
#for main_angle in main_angles:
    # Find indices of angles close to the main angle
 #   indices = np.where(np.isclose(angles, main_angle, atol=1.0))[0]
    # Calculate the mean of angles close to the main angle
  #  mean_angle = np.mean(angles[indices])
   # means.append(mean_angle)

# Convert main_angles and means to degrees
#main_angles_deg = np.rad2deg(main_angles)
#means_deg = np.rad2deg(means)

#print("Main angles (degrees):", main_angles_deg)
#print("Means of angles close to the main angles (degrees):", means_deg)
#////////////////////////////////////////////////////

# Rotate the image clockwise by 90 degrees
rotated_image = rotate_clockwise(img, 180-main_angles[1])
rotated_image2=rotate_clockwise(img, 90-main_angles[0])
# Plot the input image and the detected horizontal lines
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
ax = axes.ravel()

ax[0].imshow(rotated_image, cmap=cm.gray)  # Display the rotated image
ax[0].set_title('Rotated image')
ax[0].set_axis_off()

ax[1].imshow(edges2, cmap=cm.gray)
ax[1].set_ylim((edges2.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('Detected horizontal lines')


ax[2].imshow(rotated_image2, cmap=cm.gray)  # Display the rotated image
ax[2].set_title('Rotated image 2')
ax[2].set_axis_off()
for angle, dist in zip(angles, distances):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[1].axline((x0, y0), slope=np.tan(angle + np.pi/2))

plt.tight_layout()
plt.show()

#img_blur2 = cv2.GaussianBlur(img_res,(41, 41),cv2.BORDER_DEFAULT)
#show_image(img_blur2)

#edges2 = cv2.Canny(image=img_res, threshold1=10, threshold2=10)
#show_image(edges)


#area0 = cv2.contourArea(contours[1])
#perimeter = cv2.arcLength(contours[1], False)
#print(area0)
#print(perimeter)
#print(hierarchy) #[Next, Previous, First_Child, Parent]
#print(contours[1]) 




