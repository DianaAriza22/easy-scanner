import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import io, color, filters
import matplotlib.pyplot as plt
from matplotlib import cm
import glob

# Input path to the folder containing images
#image_folder = "C:/Users/rugel/OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER/UNIVERSIDAD/LEARNING PYTHON/ProyectoCV/scanaugm/"

# Retrieve a list of image file paths in the folder
#image_paths = glob.glob(image_folder + "*.jpeg")  # Update the file extension if necessary
image_path = "data/data(42).jpg"


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


# Plot the input image and the detected horizontal lines
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(edges_h, cmap=cm.gray)
ax[1].set_ylim((edges_h.shape[0], 0))
ax[1].set_axis_off()
ax[1].set_title('Detected horizontal lines')

for _, angle, dist in zip(*hough_line_peaks(h_h, theta_h, d_h)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[1].axline((x0, y0), slope=np.tan(angle + np.pi/2))

plt.tight_layout()
plt.show()
