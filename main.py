import cv2
import numpy as np

#https://stackoverflow.com/questions/56604151/how-to-extract-multiple-objects-from-an-image-using-python-opencv

def howGray(imageinput):
# Convert the image to grayscale
    gray = cv2.cvtColor(imageinput, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to separate gray pixels
    threshold_value = 128
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Count the gray pixels
    gray_pixel_count = np.count_nonzero(thresholded)

    # Calculate the percentage of gray pixels
    total_pixels = gray.shape[0] * gray.shape[1]
    gray_percentage = (gray_pixel_count / total_pixels) * 100

    return gray_percentage

# Load image, grayscale, Gauss
# ian blur, Otsu's threshold, dilate
image = cv2.imread('./onething.png')
original = image.copy() 
ogHeight, ogWidth, channelNumbner = image.shape
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=2)

# Find contours, obtain bounding box coordinates, and extract ROI
cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
image_number = 0
for c in cnts:
    
    x,y,w,h = cv2.boundingRect(c)

    #remove really small image
    if w < 50 and h < 50:
        continue
    if (w > (.5 *ogHeight)) or (h >(.5*ogWidth)):
        continue
    # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h+14, x:x+w+14]
    print(x,y,w,h)
    print("how gray:", howGray(ROI))
    if howGray(ROI) < 1.0:
        continue
    ROI = original[y-14:y+h+14, x-14:x+w+14]
    cv2.rectangle(image, (x-14, y-14), (x + w + 14, y + h + 14), (36,255,12), 2)
    cv2.imwrite("ROI_{}.png".format(image_number), ROI)
    image_number += 1

cv2.imshow('image', image)
# cv2.imshow('thresh', thresh)
# cv2.imshow('dilate', dilate)
cv2.waitKey()