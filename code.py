import cv2
import numpy as np
import streamlit as st

# Function for log transformation
def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_transformed = c * (np.log(img + 1))
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    return log_transformed

# Function for gamma correction
def gamma_correction(img, gamma=1.0):
    gamma_corrected = np.uint8(((img / 255.0) ** gamma) * 255)
    return gamma_corrected

# Function for contrast stretching
def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    contrast_stretched = np.interp(img, (p2, p98), (0, 255))
    contrast_stretched = np.clip(contrast_stretched, 0, 255).astype(np.uint8)  # Normalize values to [0, 255]
    return contrast_stretched

# Function for gray level slicing
def gray_level_slicing(img, low, high):
    gray_level_sliced = np.where((img >= low) & (img <= high), 255, 0)
    return gray_level_sliced

# Function for bit plane slicing
def bit_plane_slicing(img, bit_plane):
    bit_plane_sliced = (img >> bit_plane) & 1
    bit_plane_sliced *= 255
    return bit_plane_sliced

# Function for negative image
def negative_image(img):
    negative_img = 255 - img
    return negative_img

# Function for histogram equalization
def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# Function for logic operation: OR
def logic_or(image1, image2):
    result = cv2.bitwise_or(image1, image2)
    return result

# Function for logic operation: AND
def logic_and(image1, image2):
    result = cv2.bitwise_and(image1, image2)
    return result

# Function for subtraction
def image_subtraction(image1, image2):
    # Convert both images to grayscale if they are color images
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Resize the images to match dimensions if they are different
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Perform subtraction
    result = cv2.subtract(image1, image2)

    # Convert result to 3-channel image if it's not already
    if len(result.shape) < 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

# Function for averaging (smoothing filter)
def averaging_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

# Function for mean smoothing filter
def mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

# Function for median smoothing filter
def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# Function for Sobel edge detection
def sobel_operator(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    return cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

# Function for Prewitt operation
def apply_prewitt(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Define the Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Compute gradients in x and y directions
    prewitt_x = cv2.filter2D(gray, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray, -1, kernel_y)
    
    # Combine gradients to obtain magnitude
    prewitt = np.sqrt(prewitt_x*2 + prewitt_y*2)
    
    # Normalize values to be within [0, 255]
    prewitt = (prewitt - prewitt.min()) / (prewitt.max() - prewitt.min()) * 255
    prewitt = np.uint8(prewitt)
    
    # Convert grayscale image to 3-channel BGR
    prewitt_bgr = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
    
    return prewitt_bgr

# Function for Laplace operation
def apply_laplace(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Clip values to the range [0.0, 1.0]
    laplacian = np.clip(laplacian, 0.0, 1.0)
    
    # Convert grayscale image to 3-channel BGR
    laplacian_bgr = cv2.cvtColor(np.uint8(laplacian * 255), cv2.COLOR_GRAY2BGR)
    
    return laplacian_bgr

# Function for Robert operation
def apply_robert(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Define the Robert kernels
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    
    # Compute gradients in x and y directions
    robert_x = cv2.filter2D(gray, -1, kernel_x)
    robert_y = cv2.filter2D(gray, -1, kernel_y)
    
    # Combine gradients to obtain magnitude
    robert = np.sqrt(robert_x*2 + robert_y*2)
    
    # Clip values to the range [0.0, 1.0]
    robert = np.clip(robert, 0.0, 1.0)
    
    # Convert grayscale image to 3-channel BGR
    robert_bgr = cv2.cvtColor(np.uint8(robert * 255), cv2.COLOR_GRAY2BGR)
    
    return robert_bgr

# Function for Canny operation
def apply_canny(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Convert the single-channel binary image to a 3-channel grayscale image
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_bgr

# Function for Min-Max filtering
def apply_min_max(image):
    kernel = np.ones((3, 3), np.uint8)
    min_filtered = cv2.erode(image, kernel)
    max_filtered = cv2.dilate(image, kernel)
    return min_filtered, max_filtered

# Function for percentile filtering
def apply_percentile(image, percentile):
    img_flat = image.flatten()
    threshold = np.percentile(img_flat, percentile)
    image[image < threshold] = 0
    return image

st.title("Image Processing Operations")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    operation = st.selectbox("Select operation", [
        "Log Transformation",
        "Gamma Correction",
        "Contrast Stretching",
        "Gray Level Slicing",
        "Bit Plane Slicing",
        "Negative Image",
        "Histogram Equalization",
        "Logic OR",
        "Logic AND",
        "Image Subtraction",
        "Averaging Filter",
        "Mean Filter",
        "Median Filter",
        "Sobel Edge Detection",
        "Prewitt Operator",
        "Laplace Operator",
        "Robert Operator",
        "Canny Operation",
        "Min-Max Filtering",
        "Percentile Filtering"
    ])

    if operation == "Log Transformation":
        st.subheader("Log Transformation")
        st.image(log_transform(image), channels="BGR")

    elif operation == "Gamma Correction":
        gamma_value = st.slider("Gamma Value", 0.1, 5.0, 1.0, 0.1)
        st.subheader("Gamma Correction")
        st.image(gamma_correction(image, gamma_value), channels="BGR")

    elif operation == "Contrast Stretching":
        st.subheader("Contrast Stretching")
        st.image(contrast_stretching(image), channels="BGR")

    elif operation == "Gray Level Slicing":
        low = st.slider("Low Threshold", 0, 255, 0)
        high = st.slider("High Threshold", 0, 255, 255)
        st.subheader("Gray Level Slicing")
        st.image(gray_level_slicing(image, low, high), channels="BGR")

    elif operation == "Bit Plane Slicing":
        bit_plane = st.slider("Bit Plane", 0, 7, 0)
        st.subheader("Bit Plane Slicing")
        st.image(bit_plane_slicing(image, bit_plane), channels="BGR")

    elif operation == "Negative Image":
        st.subheader("Negative Image")
        st.image(negative_image(image), channels="BGR")

    elif operation == "Histogram Equalization":
        st.subheader("Histogram Equalization")
        st.image(histogram_equalization(image), channels="BGR")

    elif operation == "Logic OR":
        st.subheader("Logic OR")
        uploaded_file2 = st.file_uploader("Choose another image...", type=["jpg", "png"])
        if uploaded_file2 is not None:
            image2 = np.array(bytearray(uploaded_file2.read()), dtype=np.uint8)
            image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
            st.image(logic_or(image, image2), channels="BGR")

    elif operation == "Logic AND":
        st.subheader("Logic AND")
        uploaded_file2 = st.file_uploader("Choose another image...", type=["jpg", "png"])
        if uploaded_file2 is not None:
            image2 = np.array(bytearray(uploaded_file2.read()), dtype=np.uint8)
            image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
            st.image(logic_and(image, image2), channels="BGR")

    elif operation == "Image Subtraction":
        st.subheader("Image Subtraction")
        uploaded_file2 = st.file_uploader("Choose another image...", type=["jpg", "png"])
        if uploaded_file2 is not None:
            image2 = np.array(bytearray(uploaded_file2.read()), dtype=np.uint8)
            image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
            st.image(image_subtraction(image, image2), channels="BGR")

    elif operation == "Averaging Filter":
        kernel_size = st.slider("Kernel Size", 3, 21, 3, step=2)
        st.subheader("Averaging Filter")
        st.image(averaging_filter(image, kernel_size), channels="BGR")

    elif operation == "Mean Filter":
        kernel_size = st.slider("Kernel Size", 3, 21, 3, step=2)
        st.subheader("Mean Filter")
        st.image(mean_filter(image, kernel_size), channels="BGR")

    elif operation == "Median Filter":
        kernel_size = st.slider("Kernel Size", 3, 21, 3, step=2)
        st.subheader("Median Filter")
        st.image(median_filter(image, kernel_size), channels="BGR")

    elif operation == "Sobel Edge Detection":
        st.subheader("Sobel Edge Detection")
        st.image(sobel_operator(image), channels="BGR")

    elif operation == "Prewitt Operator":
        st.subheader("Prewitt Operation")
        st.image(apply_prewitt(image), channels="BGR")

    elif operation == "Laplace Operator":
        st.subheader("Laplace Operation")
        st.image(apply_laplace(image), channels="BGR")

    elif operation == "Robert Operator":
        st.subheader("Robert Operation")
        st.image(apply_robert(image), channels="BGR")

    elif operation == "Canny Operation":
        st.subheader("Canny Operation")
        st.image(apply_canny(image), channels="BGR")

    elif operation == "Min-Max Filtering":
        st.subheader("Min-Max Filtering")
        min_filtered, max_filtered = apply_min_max(image)
        st.image(min_filtered, channels="BGR", caption="Min Filtered")
        st.image(max_filtered, channels="BGR", caption="Max Filtered")

    elif operation == "Median Filtering":
        st.subheader("Median Filtering")
        st.image(apply_median(image), channels="BGR")

    elif operation == "Percentile Filtering":
        percentile = st.slider("Percentile", 0, 100, 50)
        st.subheader("Percentile Filtering")
        st.image(apply_percentile(image, percentile), channels="BGR")




st.title("Image Processing Operations in combo")
print("you can apply whichever operation you want in whichever order:")
uploaded = st.file_uploader("Choose an image...", type=["jpg", "png"], key="uploader1")


# Function to apply selected operations to the image
def apply_operations(image, operations):
    result = image.copy()
    for operation in operations:
        if operation == "Prewitt Operator":
            result = apply_prewitt(result)
        elif operation == "Laplace Operator":
            result = apply_laplace(result)
        elif operation == "Robert Operator":
            result = apply_robert(result)
        elif operation == "Canny Operation":
            result = apply_canny(result)
    return result

if uploaded is not None:
    image = np.array(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    operations = st.multiselect("Select operations to apply", [
        "Prewitt Operator",
        "Laplace Operator",
        "Robert Operator",
        "Canny Operation"
    ])

    if len(operations) > 0:
        result = apply_operations(image, operations)
        st.subheader("Result")
        st.image(result,channels="BGR")
