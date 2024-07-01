from PIL import Image
import pytesseract
import cv2
import numpy as np


# # Path to the tesseract executable installed via Conda
pytesseract.pytesseract.tesseract_cmd = r'/Users/jmdelahanty/miniforge3/envs/fish_control/bin/tesseract'  # Update if your path is different

# # Open an image file
# image_path = 'IMG_3273.jpg'  # Replace with your image file path
# try:
#     img = Image.open(image_path)
#     img = img.convert('RGB')  # Convert image to RGB format

#     # Perform OCR on the image with custom configuration
#     text = pytesseract.image_to_string(img)

#     # Print the extracted text
#     print("Extracted Text: ", text)

# except Exception as e:
#     print(f"Error: {e}")

# import pytesseract
# import cv2
# import numpy as np

# def preprocess_image(image, method='canny'):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
#     if method == 'threshold':
#         return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     elif method == 'adaptive':
#         return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     elif method == 'canny':
#         return cv2.Canny(gray, 100, 200)

def improve_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

# def detect_lines(image):
#     edges = cv2.Canny(image, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
#     line_mask = np.zeros(image.shape, dtype=np.uint8)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)
#     return line_mask

# img = cv2.imread('IMG_3273.jpg')
# improved_image = improve_contrast(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# line_mask = detect_lines(improved_image)
# text_regions = cv2.bitwise_and(improved_image, improved_image, mask=cv2.bitwise_not(line_mask))

# # Multiple preprocessing methods
# preprocessed_images = [
#     preprocess_image(text_regions, 'threshold'),
#     preprocess_image(text_regions, 'adaptive'),
#     cv2.dilate(text_regions, np.ones((2,2), np.uint8), iterations=1),
#     cv2.erode(text_regions, np.ones((2,2), np.uint8), iterations=1)
# ]

# result_img = img.copy()
# all_detected_text = set()

# for idx, processed_img in enumerate(preprocessed_images):
#     custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789>#/'
#     data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)

#     for i, word in enumerate(data['text']):
#         if word.strip():
#             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#             cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             all_detected_text.add(word.strip())

# print("All detected text:")
# for text in all_detected_text:
#     print(text)

# cv2.imshow('Detected text', result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def isolate_white_tag(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for white color in HSV
    lower_white = np.array([0, 0, 155])
    upper_white = np.array([180, 100, 255])
    
    # Create a mask for white color
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((21,21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the tag)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Create a mask for the largest contour
        tag_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(tag_mask, [largest_contour], 0, (255), -1)
        
        # Extract the tag area
        tag_area = cv2.bitwise_and(image, image, mask=tag_mask)
        
        return tag_area[y:y+h, x:x+w]
    
    return None

def denoise_image(image):
    # Apply fastNlMeansDenoising
    denoised = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21)
    return denoised

# Load the image
img = cv2.imread('IMG_3273.jpg')

# Isolate the white tag
tag_image = isolate_white_tag(img)

cv2.imshow('Original Image', tag_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


if tag_image is not None:
    # Apply your existing preprocessing and OCR on the tag_image
    gray = cv2.cvtColor(tag_image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    denoised = denoise_image(sharpen)
    threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    denoised = denoise_image(threshold)
    # Perform OCR
    custom_config = r'--oem 3 --psm 12'
    text = pytesseract.image_to_string(denoised, config=custom_config)

    print("Extracted text:")
    print(text, end='\n')
    # Display the isolated tag
    cv2.imshow('Isolated Tag', tag_image)
    cv2.imshow('Threshold', threshold)
    cv2.imshow('Denoised', denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not isolate the white tag")
