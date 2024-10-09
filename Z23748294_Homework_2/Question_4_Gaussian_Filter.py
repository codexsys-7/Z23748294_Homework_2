import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
import os
import cv2

# Load the image
img = cv2.imread('triangles.jpeg', 0)

# Applying the Fourier Transform to the image to apply butterworth filter.
f = np.fft.fft2(img) # Convert the image to the frequency domain.
f_shift = np.fft.fftshift(f) # Shifting the frequency domain center to make the analysis and applying filters easier.

# Gaussian filter using the formula.
def gaussian_filter_1(shape, sigma):
    A_rows, Q_cols = shape
    # u_cen_rows and v_cen_cols representing the center indices of the rows and columns,
    u_cen_rows = np.arange(A_rows) - A_rows/2
    v_cen_cols = np.arange(Q_cols) - Q_cols/2
    U, V = np.meshgrid(v_cen_cols, u_cen_rows) 
    # Calculate the distance from the center of the image using euclidean distance function.
    D_uv = np.sqrt(U ** 2 + V ** 2)  
    mask = np.exp(-(D_uv**2) / (2 * sigma**2)) # Gaussian formula
    return mask

sigma = 20  # Changing the sigma value will change the intensity of the filter, higher or lower.
# Call the function with given parametres, Image shape and sigma value.
gaussian_inputs = gaussian_filter_1(img.shape, sigma)



# Apply the Gaussian filter using formula
filtered_f_transform = f_shift * gaussian_inputs
# Inverse Fourier Transform to get the filtered image.
# computing the inverse 2D Fast Fourier Transform (FFT) to convert the filtered frequencies domain back to the original image.
f_ishift = np.fft.ifftshift(filtered_f_transform)
gaussian_img = np.fft.ifft2(f_ishift)
# The np.abs() function extracts the final values which can be used to display the image.
gaussian_img = np.abs(gaussian_img)
# Save the filtered image
# file_name = 'gaussian_filtered_image.jpg'
# Image.fromarray(np.uint8(gaussian_img)).save(file_name)



# Applying gaussian filter using the in-built library.
gaussian_library_image = gaussian_filter(np.abs(f_shift), sigma)
# Apply the Gaussian filter
gaussian_library_filtered_f_transform = f_shift * gaussian_library_image
# Inverse Fourier Transform to get the filtered image.
# computing the inverse 2D Fast Fourier Transform (FFT) to convert the filtered frequencies domain back to the original image.
gaussian_library_f_ishift = np.fft.ifftshift(gaussian_library_filtered_f_transform)
gaussian_library_img = np.fft.ifft2(gaussian_library_f_ishift)
# The np.abs() function extracts the final values which can be used to display the image.
gaussian_library_img = np.abs(gaussian_library_img)
# Save the filtered image
# file_name_2 = 'gaussian_library_filtered_image.jpg'
# Image.fromarray(np.uint8(gaussian_library_img)).save(file_name_2)




# Display the original and Gaussian filtered images
plt.figure(figsize=(15, 5))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Gaussian Formula Filtered Image')
plt.imshow(gaussian_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Gaussian Library Filtered Image')
plt.imshow(gaussian_library_img, cmap='gray')
plt.savefig('Gaussian_Filter_Question_4.jpg', bbox_inches='tight', pad_inches=0)
plt.axis('off')


plt.tight_layout()
plt.show()
