# digital-image-processing-toolkit
# Digital Image Processing Toolkit

This project is a Digital Image Processing Toolkit implemented in Python using OpenCV and Streamlit. The toolkit provides an easy-to-use interface for performing a variety of image processing operations on digital images.

## Overview

The toolkit leverages Streamlit, a Python library for building interactive web applications, to create a user-friendly interface. Users can upload their images and select from a range of image processing operations to apply.

## Features

### Interface with Streamlit

The toolkit utilizes Streamlit to create a web-based interface, making it accessible through a web browser. Streamlit provides an intuitive and interactive environment for users to upload images and interact with various image processing functions.

### Image Processing Operations

The toolkit offers a comprehensive set of image processing operations, including:

- **Log Transformation**: Adjusts the dynamic range of an image using logarithmic function.
- **Gamma Correction**: Adjusts the brightness and contrast of an image using gamma correction.
- **Contrast Stretching**: Enhances the contrast of an image by stretching the intensity range.
- **Gray Level Slicing**: Highlights specific intensity ranges in an image.
- **Bit Plane Slicing**: Extracts specific bit planes from the pixel values of an image.
- **Negative Image**: Computes the negative of an image.
- **Histogram Equalization**: Enhances the contrast of an image by equalizing its histogram.
- **Logic Operations (OR and AND)**: Performs logical operations on two images.
- **Image Subtraction**: Computes the difference between two images.
- **Smoothing Filters (Averaging, Mean, and Median)**: Applies spatial filters to remove noise from images.
- **Edge Detection (Sobel, Prewitt, Laplace, Robert, and Canny)**: Detects edges in images using various edge detection algorithms.
- **Min-Max and Percentile Filtering**: Applies morphological operations for noise reduction and feature extraction.

### Customizable Sequence of Operations

One unique feature of the toolkit is the ability for users to apply multiple image processing operations in their preferred sequence. Users can select and combine operations to create personalized image processing pipelines tailored to their specific needs.

## Usage

To use the toolkit, simply run the Streamlit application by executing the main script. Users can then upload their images, select desired operations from the dropdown menu, and interact with the processed images in real-time.

## Contributors

- Vaishnavi Dave(https://github.com/vaish414)
- Drashti Joshi()
- Taneeshk Patel()

## Contributions

Contributions to the project are welcome! Whether it's adding new image processing operations, improving existing functionalities, or enhancing the user interface, your contributions can help make the toolkit more robust and versatile. Feel free to open issues or submit pull requests on GitHub to contribute to the project.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for both commercial and non-commercial purposes, subject to the terms of the license.
