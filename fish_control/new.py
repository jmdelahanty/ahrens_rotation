import cv2
import numpy as np
import pytesseract

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine horizontal and vertical lines
    table_lines = cv2.addWeighted(detect_horizontal, 1, detect_vertical, 1, 0)
    
    return table_lines

def find_contours(image, table_lines):
    # Dilate the lines to connect nearby components
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(table_lines, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    min_area = 100  # Adjust this value based on your image
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return filtered_contours

def create_rois(image, contours):
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Add padding
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        roi = image[y:y+h, x:x+w]
        rois.append((x, y, w, h, roi))
    return rois

def preprocess_roi(roi):
    if roi.size == 0:
        return None
    
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, searchWindowSize=21)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    except cv2.error:
        return None

# Load image
image = cv2.imread('test.png')

if image is None:
    print("Error: Could not read the image file.")
    exit()

# Detect lines
table_lines = detect_lines(image)

# Find contours (cells)
contours = find_contours(image, table_lines)

# Create ROIs
rois = create_rois(image, contours)

# Process each ROI
for i, (x, y, w, h, roi) in enumerate(rois):
    processed_roi = preprocess_roi(roi)
    
    if processed_roi is not None:
        # Perform OCR on the processed ROI
        text = pytesseract.image_to_string(processed_roi, config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#>')
        
        print(f"Text in region {i} ({w}x{h}): {text.strip()}")
        
        # Draw the bounding box on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Detected Regions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()