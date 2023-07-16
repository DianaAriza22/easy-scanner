import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import io, color, filters
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

# Function to rotate an image clockwise
def rotate_clockwise(image, degrees):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -degrees, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# Input path to the single image
image_path = "data/data(29).jpg"

# Load the source image
image = io.imread(image_path)

# Convert the image to grayscale
image_gray = color.rgb2gray(image)

# Apply Gaussian blur to reduce noise
image_blurred = filters.gaussian(image_gray, sigma=1.5)

# Apply horizontal Sobel filter
image_edges_h = filters.sobel_h(image_blurred)

# Perform Canny edge detection on horizontal Sobel output
edges_h = canny(image_edges_h, sigma=1.8)

# Perform Hough transform on the horizontal edges
tested_angles_h = np.linspace(- np.pi /2, np.pi / 2, 360, endpoint=False)
h_h, theta_h, d_h = hough_line(edges_h, theta=tested_angles_h)

# Extract prominent lines from the Hough accumulator using a threshold
_, angles, distances = hough_line_peaks(h_h, theta_h, d_h, threshold=600)
print(np.rad2deg(angles))
# Perform K-means clustering to find the main angles
if len(angles) > 1:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(angles.reshape(-1, 1))
    main_angles = kmeans.cluster_centers_.flatten()
else:
    main_angles = angles
print('testing main  ',np.rad2deg(main_angles))
main_angles=np.rad2deg(main_angles)
# Calculate the means of angles close to the main angles
means = []
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

# Rotate the image clockwise by 90 degrees
rotated_image = rotate_clockwise(image, 180-main_angles[1])
rotated_image2=rotate_clockwise(image, 90-main_angles[0])
# Plot the input image and the detected horizontal lines
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
ax = axes.ravel()

ax[0].imshow(rotated_image, cmap=cm.gray)  # Display the rotated image
ax[0].set_title('Rotated image')
ax[0].set_axis_off()

ax[1].imshow(edges_h, cmap=cm.gray)
ax[1].set_ylim((edges_h.shape[0], 0))
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
