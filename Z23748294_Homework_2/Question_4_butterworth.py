import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Loading the image using opencv.
image = cv2.imread('triangles.jpeg', 0) 

# Applying the Fourier Transform to the image to apply butterworth filter.
f = np.fft.fft2(image) # Convert the image to the frequency domain.
fshift = np.fft.fftshift(f) # Shifting the frequency domain center to make the analysis and applying filters easier.

# Butterworth Filter
def butterworth_filter(shape, cutoff, n):
    A_rows, Q_cols = shape
    # u_cen_rows and v_cen_cols representing the center indices of the rows and columns,
    u_cen_rows = np.arange(A_rows) - A_rows/2
    v_cen_cols = np.arange(Q_cols) - Q_cols/2
    U, V = np.meshgrid(v_cen_cols, u_cen_rows) 
    # Calculate the distance from the center of the image using euclidean distance function.
    D_uv = np.sqrt(U ** 2 + V ** 2)  
    # Butterworth filter coefficeints are calculated using distance values and cutoff value.
    H = 1 / (1 + (D_uv / cutoff) ** (2 * n))  # Butterworth filter formula
    return H


# Apply Butterworth filter
cutoff = 30  # Adjust the cutoff values as needed.
n = 2  # Change the order for different responses.


# Call the function with given parametres, Image shape, cutoff value and order (n).
butterworth_inputs = butterworth_filter(image.shape, cutoff, n)
# Apply the filter in frequency domain, using the fshift value and butterworth functions result.
fshift_filtered = fshift * butterworth_inputs
# Inverse Fourier Transform to get the filtered image
f_ishift = np.fft.ifftshift(fshift_filtered) # Applying inverse shift to the filtered frequeny domain, shift the frequency componets back to their original locations.
# computing the inverse 2D Fast Fourier Transform (FFT) to convert the filtered frequencies domain back to the original image.
butterworth_img = np.fft.ifft2(f_ishift)
# The np.abs() function extracts the final values which can be used to display the image.
butterworth_img = np.abs(butterworth_img)
# Save the filtered image
# butterworth_filtered_image_path = 'butterworth_filtered_image.jpg'
# Image.fromarray(np.uint8(butterworth_img)).save(butterworth_filtered_image_path)





# Plot the original and the filtered images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Butterworth Filter Result
plt.subplot(2, 2, 2)
plt.imshow(butterworth_img, cmap='gray')
plt.title('Butterworth Filtered Image')
plt.savefig('Butterworth_Question_4_filter.jpg', bbox_inches='tight', pad_inches=0)
plt.axis('off')

plt.tight_layout()
plt.show()