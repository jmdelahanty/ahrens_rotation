import cv2
import numpy as np

def improve_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def detect_vlines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve Contrast
    gray = improve_contrast(gray)
    
    # Apply adaptive threshold
    image_thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 0)

    # Apply morphological opening with a vertical line kernel
    kernel = np.ones((image.shape[0] // 20, 1), dtype=np.uint8)  # Adjust the size of the kernel as needed
    image_mop = cv2.morphologyEx(image_thr, cv2.MORPH_OPEN, kernel)
    min_width = 10  # Minimum width of the horizontal line to keep
    min_height = 100  # Minimum height of the horizontal line to keep
    # Canny edge detection
    image_canny = cv2.Canny(image_mop, 50, 150)  # Adjust the Canny parameters as needed
    filtered_vcnts = []
    for c in vcnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_width and h > min_height:
            filtered_vcnts.append(c)
            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(image_canny, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    return lines, image_thr, image_mop, image_canny


def detect_hlines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = improve_contrast(gray)
    
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Apply adaptive threshold
    # image_thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 0)

    # Apply morphological closing with a horizontal line kernel to enhance horizontal lines
    kernel = np.ones((1, image.shape[1] // 20), dtype=np.uint8)  # Adjust the size of the kernel as needed
    image_morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Canny edge detection
    image_canny = cv2.Canny(image_morph, 50, 150)  # Adjust the Canny parameters as needed

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(image_canny, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=100)
    
    return lines, thresh, image_morph, image_canny

def draw_lines(image, lines, color=(0, 255, 0)):
    for (x1, y1, x2, y2) in lines:
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

# Load the image
image_path = 'test.png'
image = cv2.imread(image_path)

# Detect vertical lines in the image
vlines, v_thr, v_morph, v_canny = detect_vlines(image)

# Detect horizontal lines in the image
hlines, h_thr, h_morph, h_canny = detect_hlines(image)


# (Visualization) Output
cv2.imshow('Original Image', image)
cv2.imshow('Vertical Thresholded Image', v_thr)
cv2.imshow('Vertical Morphologically Opened Image', v_morph)
cv2.imshow('Vertical Canny Edge Detection', v_canny)
cv2.imshow('Horizontal Thresholded Image', h_thr)
cv2.imshow('Horizontal Morphologically Enhanced Image', h_morph)
cv2.imshow('Horizontal Canny Edge Detection', h_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
