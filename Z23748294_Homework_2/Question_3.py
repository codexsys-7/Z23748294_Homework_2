"""
3. Explain what a Kuwahara filter is, and apply it to the image using either Python or MATLAB to demonstrate its effect.
The Kuwahara filter is an image smoothing tool that helps reduce noise while keeping edges sharp. 
It works by splitting the surrounding area into overlapping sections, typically four quadrants. 
The filter calculates the average and variance of pixel values in each section and selects the one with the lowest variance to set the new pixel value. 
This method smooths out flat areas while preserving the edges or avoiding the blurring effect across the boundaries. 
It is commonly used in creating artistic images, such as pastel paintings, and in medical image analysis which is very important in preserving the boundaries around the image or an object.

"""


import cv2
import numpy as np
from scipy.ndimage import uniform_filter

# the overlapping_regions represents the four regions aroung each pixel in an image.
overlapping_regions = [(0, 0), (0, 1), (1, 0), (1, 1)]

def Kuwa_Func(image, kernel_size=5):
    # the kernel size is used to determine the size of the neighboring pixels centered around each pixel.
    radius = kernel_size // 2
    # # If the given image is a grayscale image.
    if len(image.shape) == 2:
        print("Image Shape is:", image.shape)
        print("Image is a GrayScale")
        # Apply padding to the image with a reflection of the image's boundary. The amount of padding is determined by the 'radius' parameter.
        padding_img = np.pad(image, pad_width=radius, mode='reflect')
        final_image = np.zeros_like(image) #  Create an empty array of the same shape as the original image to store the final result.
        gray_means = np.zeros((4, image.shape[0], image.shape[1])) # Initialize a 4D array to store the mean values of the grayscale image for four directions.
        # The shape of this array matches the original image, with 4 additional slices for different directions.
        gray_var = np.zeros((4, image.shape[0], image.shape[1])) # Similar to gray_means, this array holds variance values across the four directions.


        for i, (dx, dy) in enumerate(overlapping_regions):
            shifted_image = padding_img[dx:dx + image.shape[0], dy:dy + image.shape[1]]
            # Calcualting the mean and variance pixel value within a certain neighborhood using the uniform filter fucntion, which helps in smoothing the image.
            gray_means[i] = uniform_filter(shifted_image, size=kernel_size)
            gray_var[i] = uniform_filter(shifted_image*2, size=kernel_size) - gray_means[i]*2

        # Selecting the means corresponding to the smallest variance.
        min_var = np.argmin(gray_var, axis=0)
        output = np.choose(min_var, gray_means)
        final_image = output.astype(image.dtype)
    
    else:  # if the given image is an RGB format.
        print("Image is RGB Format.")
        print("Image Shape is:", image.shape)
        # Apply padding to the image with a reflection of the image's boundary. The amount of padding is determined by the 'radius' parameter.
        padding_img = np.pad(image, pad_width=((radius, radius), (radius, radius), (0, 0)), mode='reflect')
        final_image = np.zeros_like(image)
        for j in range(3):  # Applying the function to each Channel of the image seperately.
            rgb_means = np.zeros((4, image.shape[0], image.shape[1]))
            rgb_var = np.zeros((4, image.shape[0], image.shape[1]))

            for i, (dx, dy) in enumerate(overlapping_regions):
                shifted_image = padding_img[dx:dx + image.shape[0], dy:dy + image.shape[1], j]
                rgb_means[i] = uniform_filter(shifted_image, size=kernel_size)
                rgb_var[i] = uniform_filter(shifted_image*2, size=kernel_size) - rgb_means[i]*2

            # Selecting the means corresponding to the smallest variance.
            rgb_min_var = np.argmin(rgb_var, axis=0)
            output = np.choose(rgb_min_var, rgb_means)
            final_image[:, :, j] = output.astype(image.dtype)

    return final_image


# Loading the image using cv2.imread().
image = cv2.imread('triangles.jpeg')

# Call the function with given parametres, Image and Kernel size.
kuwafilter_image = Kuwa_Func(image, kernel_size=5)

# Saving the image using Imwrite fucntion.
# Click on ESC key to exit the image window.
cv2.imwrite('triangles_smoothed.jpg', kuwafilter_image)
cv2.imshow('Kuwahara Filter.', kuwafilter_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

