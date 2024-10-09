"""
Dithering is a process of creating a vibrant color illusion by making use of limited color pallete. 
The process is achieved by intelligently distributing the noise around the pixels, which is mainly used when converting the images from higher color depth to lower color depth.

Comparision Results:
Floyd-Steinberg Dithering Kernel.
- The application of the Floyd-Steinberg dithering algorithm results in an image that exhibits noticeble pixalation compared to the original. 
- This technique effectively redistributes quantization errors to adjacent pixels, leading to a slightly grainy texture. 
- Nevertheless, the majority of the image's details are retained, presenting a more textured quality as a consequence of the error diffusion process.

Jarvis-Judice-Ninke Dethering Kernel.
- The implementation of the Jarvis Judice Ninke Kernel has enhanced the image quality, resulting in a finer appearance characterized by a pixelated effect attributed to the larger kernel size. 
- This image exhibits increased vibrancy, reduced graininess, and a slightly softer texture in regions with delicate color transitions when compared to the Floyd-Steinberg method. 
- However, a significant drawback of this dithering technique is its computational expensive, as it employs a 3x5 kernel size, which is considerably larger than that used in the Floyd-Steinberg approach.

Conclusion:
Ultimately, both dithering algorithms perform effectively, yet they vary in their methods of error diffusion, resulting in these subtle visual distinctions.
"""



import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
image = cv2.imread('triangles.jpeg')
print("Shape of the image:", image.shape)


# Floyd-Steinberg dithering kernel
floyd_kernel = np.array([[0, 0, 7], [3, 5, 1]]) / 16

# Jarvis-Judice-Ninke dithering kernel
jarvis_kernel = np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]) / 48

def dithering_function(image, kernel_size):
    output_img = image.copy()
    height, width, channels = image.shape

    # Applying the dithering effects to each color Channel.
    for i in range(channels):
        for h in range(height):
            for w in range(width):
                opix = output_img[h, w, i]
                # print("My Old Pixel value:", opix)
                npix = np.round(opix / 255.0) * 255.0 # Quantizing the pixel values which are rounded up to either 0 or 255 (Binary colors) for the effect.
                output_img[h, w, i] = npix # mapping quantized new pixel value to the output image.
                quant_error = opix - npix # Calcualting the differecene between old pixel and quantized pixel

                # Distribute the error to the neighboring pixels using the given kernel to apply the dithering effect.
                for dy, row in enumerate(kernel_size):
                    # print("My (dy, row)", dy, row)
                    for dx, kernel_value in enumerate(row):
                        nx, ny = w + dx - 1, h + dy
                        # Ensuring the coordinates are not out of bounds when distributing the errors.
                        if (0 <= nx < width) and (0 <= ny < height):
                            # Distribute the erroe to the neighboring pixels.
                            output_img[ny, nx, i] += quant_error * kernel_value
    print("My Final Output Image:", output_img)
    return np.clip(output_img, 0, 255).astype(np.uint8) # After the quantisation is done to all the pixels, we clip the values to the original output image in the range [0-255] color intensities and use np.uint8 to extract the proper image format.



# Calling the function with Floyd-Steinberg kernel.
floyd_steinberg_result = dithering_function(image, floyd_kernel)


# Calling the function with Jarvis-Judice-Ninke kernel.
jarvis_judice_ninke_result = dithering_function(image, jarvis_kernel)





# Display the original and dithered images for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(floyd_steinberg_result)
plt.title('Floyd-Steinberg Dithering Effect Image')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(jarvis_judice_ninke_result)
plt.title('Jarvis-Judice-Ninke Dithering Effect Image')
plt.savefig('Dithering_Question_2.jpg', bbox_inches='tight', pad_inches=0)
plt.axis('off')


plt.show()