# **Image Downscaling with Custom Convolution**  

This repository contains a Python implementation of a **custom convolution-based image downscaler**. The code demonstrates how to apply a convolution operation manually (without relying heavily on libraries like OpenCV or SciPy) to reduce the resolution of an image while preserving its features.  

---

## **Overview**  
The project consists of two main functions:  
1. **`downscaler_kernel()`** → Generates a custom kernel for downscaling.  
2. **`convolution()`** → Applies the kernel to an image using convolution.  

The goal is to **reduce image resolution** while maintaining structural integrity, simulating how some image processing pipelines downscale images before further analysis.  

---

## **How It Works**  

### **1. `downscaler_kernel(size: int)`**  
- Creates a **square kernel** of size `size × size`.  
- The kernel has a **central value of 1** (to preserve the main pixel) and **0.35 in surrounding positions** (to softly blend neighboring pixels).  
- Example for `size=5`:  
  ```
  [[0.35, 0.35, 0.35, 0.35, 0.35],
   [0.35, 0.35, 0.35, 0.35, 0.35],
   [0.35, 0.35,   1,  0.35, 0.35],
   [0.35, 0.35, 0.35, 0.35, 0.35],
   [0.35, 0.35, 0.35, 0.35, 0.35]]
  ```  

### **2. `convolution(image, kernel: list, step: int)`**  
- Applies the kernel to the input image using **discrete convolution**.  
- **Parameters:**  
  - `image`: Input grayscale image (NumPy array).  
  - `kernel`: The convolution kernel (2D list).  
  - `step`: Controls downscaling stride (higher `step` → more aggressive downscaling).  
- **How it works:**  
  - Iterates over the image, skipping pixels based on `step`.  
  - For each position, computes a weighted sum of the kernel and the corresponding image region.  
  - Stores the result in a new downscaled image.  

---

## **Usage**  

### **1. Requirements**  
- Python 3.x  
- OpenCV (`cv2`)  
- NumPy (`numpy`)  
- Matplotlib (`matplotlib`)  

Install dependencies:  
```sh
pip install opencv-python numpy matplotlib
```

### **2. Running the Code**  
1. Place an image (e.g., `mao.jpeg`) in the same directory.  
2. Run:  
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Load image in grayscale
image = cv2.imread("mao.jpeg", cv2.IMREAD_GRAYSCALE)

# Generate kernel and apply convolution
kernel = downscaler_kernel(5)  # 5x5 kernel
downscaled_image = convolution(image, kernel, step=5)

# Display results
plt.imshow(downscaled_image, cmap='gray')
plt.show()
```

### **3. Expected Output**  
- The original image is **downscaled** while preserving edges and structures.  
- Higher `step` values produce **smaller output images**.  

---

## **🔍 Key Concepts**  
✅ **Manual Convolution** → No reliance on `cv2.filter2D` or `scipy.signal.convolve2d`.  
✅ **Custom Kernel** → Adjust weights to control blurring/downscaling effects.  
✅ **Stride Control** → The `step` parameter allows flexible downscaling.  

---

## **📂 Repository Structure**  
```
.
├── README.md          # This file
├── downscaler.py      # Main Python script
├── mao.jpeg           # Example input image
└── requirements.txt   # Dependencies
```

---
