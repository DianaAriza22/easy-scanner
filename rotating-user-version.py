import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import io, color, filters
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

# Function to show image in window cv2
def show_image(image,window='window'):
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window,image)
    c = cv2.waitKey(0)
    if c >= 0 : cv2.destroyAllWindows()
    
# Function to rotate an image clockwise
def rotate_clockwise(image, degrees):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -degrees, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


# Read image
path = input("Image path:")
img = cv2.imread(path)
show_image(img,'Original image')

# //////////// PREPROCESSING ///////////
#applying guassian blur
img_blur = cv2.GaussianBlur(img,(5, 5),cv2.BORDER_DEFAULT)
#show_image(img_blur)

#applying Canny Filtering
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
#show_image(edges)

# Find contours
contours, hierarchy  = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find longest contour
perimeter = []
for i in range(0,len(contours)):
    perimeter.append(cv2.arcLength(contours[i], False))

sorted_indices = np.argsort(perimeter)[::-1]

largest_contours = []
number_of_contours = 5
if number_of_contours > len(perimeter):
    number_of_contours = len(perimeter)

for i in range(0,number_of_contours-1):
    largest_contours.append(contours[sorted_indices[i]])

#Show longest contours
img_blur2 = cv2.GaussianBlur(img,(41, 41),cv2.BORDER_DEFAULT) # High blur to highlight the contour
img_res = cv2.drawContours(img_blur2, largest_contours, -1, (0,0,0), 2) #(0,255,75)

img_show_contours = cv2.drawContours(img,largest_contours, -1, (0,255,0), 2)
show_image(img_show_contours, f"{number_of_contours} longest contours")

contours_check = input("Do you want to change the number of contours? (y/n): ")
if contours_check == "y" or contours_check == "Y":
    number_of_contours = int(input("New number of contours: "))
    largest_contours = []
    if number_of_contours > len(perimeter):
        number_of_contours = len(perimeter)

    for i in range(0,number_of_contours-1):
        largest_contours.append(contours[sorted_indices[i]])

    #Show longest contours
    img_blur2 = cv2.GaussianBlur(img,(41, 41),cv2.BORDER_DEFAULT) # High blur to highlight the contour
    img_res = cv2.drawContours(img_blur2, largest_contours, -1, (0,0,0), 2) #(0,255,75)

    img_show_contours = cv2.drawContours(img,largest_contours, -1, (0,255,0), 2)
    show_image(img_show_contours, f"{number_of_contours} longest contours")
else:
    pass

# RGB to B/W conversion
img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

# Appling Canny Filter
edges2 = cv2.Canny(image=img_res, threshold1=100, threshold2=200)

# Hough transform 
tested_angles_h = np.linspace(- np.pi /2, np.pi / 2, 200)
h_h, theta_h, d_h = hough_line(edges2, theta=tested_angles_h)

# Extract prominent lines from the Hough accumulator using a threshold
_, angles, distances = hough_line_peaks(h_h, theta_h, d_h, threshold=200)

# Perform K-means clustering to find the main angles
if len(angles) > 1:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(angles.reshape(-1, 1))
    main_angles = kmeans.cluster_centers_.flatten()
elif len(angles) == 1:
    main_angles = angles
else:
    main_angles = [0]
    print("It was not possible to find reference angles for the rotation")
# Convert to degrees
main_angles=np.rad2deg(main_angles)

# Rotate the image clockwise by 90 degrees
rotated_image = rotate_clockwise(img, 180-main_angles[0])
show_image(rotated_image, f"Clockwise rotation: {main_angles[0]}")

while True:
        print("\nCustom rotation:")
        print("1. Right 1x")
        print("2. Right 2x")
        print("3. Left 1x")
        print("4. Left 2x")
        print("5. Don't rotate")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            rotated_image = rotate_clockwise(img, 180-main_angles[0]-90)
            show_image(rotated_image, f"Clockwise rotation: {main_angles[0]-90}")
        elif choice == '2':
            rotated_image = rotate_clockwise(img, 180-main_angles[0]-180)
            show_image(rotated_image, f"Clockwise rotation: {main_angles[0]-180}")
        elif choice == '3':
            rotated_image = rotate_clockwise(img, 180-main_angles[0]+90)
            show_image(rotated_image, f"Clockwise rotation: {main_angles[0]}+90")
        elif choice == '4':
            rotated_image = rotate_clockwise(img, 180-main_angles[0]+180)
            show_image(rotated_image, f"Clockwise rotation: {main_angles[0]}+180")
        elif choice == '5':
            print("Exiting...")
            show_image(rotated_image, f"Clockwise rotation: {main_angles[0]}")
            break
        else:
            print("Invalid choice. Please try again.")
