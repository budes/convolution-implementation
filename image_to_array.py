import cv2 
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

image = cv2.imread("mao.jpeg", cv2.IMREAD_GRAYSCALE)

print(image.shape)
def downscaler_kernel(size: int):
    """
    size: int -> Size of the side of the kernel.
    return: kernel.

    Basically will try to create a downscaler kernel (rounded by zeros or values close to it)
    in this defined size.
    """
    
    kernel = [[0.35 for _ in range(size)] for _ in range(size)]
    kernel[size//2][size//2] = 1
    
    return kernel

def convolution(image, kernel:list, step:int):
    """
    image: np.array -> Numpy array that defines an image
    kernel: list -> List of values that works as a kernel for the convolution
    step: int -> The step applied by the movement
    
    returns: convoluted image

    Runs the convolution through the input image and applies the effect of the kernel on it.
    """
    result = [[0 for _ in range(len(kernel)//2, image.shape[1]-1, step)] for _ in range(len(kernel)//2, image.shape[0]-1, step)]

    for y in range(ceil(len(kernel)/2), image.shape[0] - ceil(len(kernel)/2), step):
        for x in range(ceil(len(kernel)/2), image.shape[1] - ceil(len(kernel)/2), step):
            index_result = 0
        
            for ky in range(-len(kernel)//2, len(kernel)//2 + 1):
                for kx in range(-len(kernel)//2, len(kernel)//2 + 1):
                    index_result += kernel[ky + 1][kx + 1] * image[y + ky][x + kx]

            index_result = round(index_result)
        
            result[(y - ceil(len(kernel)/2)) // step][(x - ceil(len(kernel)/2)) // step] = index_result
    return np.array(result)

# Showing the result
plt.imshow(convolution(image, downscaler_kernel(5), 5))
plt.show()
