import cv2
import numpy as np

def improve_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)
def find_intersections(hcnts, vcnts):
    intersections = []
    for h in hcnts:
        for v in vcnts:
            hx, hy, hw, hh = cv2.boundingRect(h)
            vx, vy, vw, vh = cv2.boundingRect(v)
            if hx < vx + vw and hx + hw > vx and hy < vy + vh and hy + hh > vy:
                ix = max(hx, vx)
                iy = max(hy, vy)
                intersections.append((ix, iy))
    return intersections
# Load image, convert to grayscale, Otsu's threshold
image = cv2.imread('test.png')
# Add a black border around the image
border_size = 10  # Size of the border
image_with_border = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
cv2.imshow('image_with_border', image_with_border)
cv2.waitKey()
cv2.destroyAllWindows()
result = image_with_border.copy()
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Improve contrast
gray = improve_contrast(gray)

# Apply thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Dilate the image
kernel = np.ones((1, 11), np.uint8)  # Adjust the kernel size as needed
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Detect horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
detect_horizontal = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
hcnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hcnts = hcnts[0] if len(hcnts) == 2 else hcnts[1]

min_width = 180  # Minimum width of the horizontal line to keep
min_height = 5  # Minimum height of the horizontal line to keep


filtered_hcnts = []
for c in hcnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > min_width and h > min_height:
        filtered_hcnts.append(c)
        cv2.drawContours(im, [c], -1, (255, 0, 0), 2)


# Detect vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
detect_vertical = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
vcnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vcnts = vcnts[0] if len(vcnts) == 2 else hcnts[1]

min_width = 10  # Minimum width of the horizontal line to keep
min_height = 100  # Minimum height of the horizontal line to keep

filtered_vcnts = []
for c in vcnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > min_width and h > min_height:
        filtered_vcnts.append(c)
        cv2.drawContours(im, [c], -1, (0, 0, 255), 2)

# Find intersections
intersections = find_intersections(filtered_hcnts, filtered_vcnts)

# Create squares from intersections
square_size = 20  # Define the size of the squares to be drawn
for (ix, iy) in intersections:
    cv2.rectangle(im, (ix, iy), (ix + square_size, iy + square_size), (0, 255, 0), 2)

cv2.imshow('result', im)
cv2.waitKey()
cv2.destroyAllWindows()
