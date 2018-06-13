# Fracture-Detection
Building a Computer Vision based tool for detecting fractures and fatiguing in mechanical components.

### Introduction
- This project aims to develop a tool for identifying fractures and fissures in a mechanical componenet.

- The tool makes use of OpenCV and TensorFlow. OpenCV is used to visually detect the presence of the fracture and TensorFlow is used to predict the presence of fractures.

#### Note:
The OpenCV code was written by the author, most of the TensorFlow code has been pulled from the TensorFlow repository.

### Working:
1. A image is sent to the OpenCV code which runs it through a series of "Kernels", which include:

	- Sobel-X
	- Sobel-Y
	- Small Blur
	- Large Blur
	- Sharpen
	- Laplacian

2. The OpenCV code serves as a detector for fractures and relays it to the operator.

3. The image is then passed to the label_image.py [script](https://github.com/SarthakJShetty/Fracture-Detection/blob/master/label_image.py), which predicts whether the object is classified as fractured or not.

4. A retrain.py [script](https://github.com/SarthakJShetty/Fracture-Detection/blob/master/retrain.py) code is provided which is trained on the dataset of images. 

5. A webscraper has been developed which scrapes Google Images for the images to build your dataset (yet to be developed).

### Results of Kerneler:

- **Laplacian Kernel:** 

<p align="center">
<img src="https://github.com/SarthakJShetty/Fracture-Detection/blob/master/Results/Laplacian.jpg"
alt="Laplacian Kerneling results"/>
</p>

- **Large Bluring:** 

<p align="center">
<img src="https://github.com/SarthakJShetty/Fracture-Detection/blob/master/Results/Large%20Bluring.jpg"
alt="Large Blurring results"/>
</p>

- **Small Bluring:** 

<p align="center">
<img src="https://github.com/SarthakJShetty/Fracture-Detection/blob/master/Results/Small%20Bluring.jpg"
alt="Small Bluring results"/>
</p>

- **Sharpen:** 	

<p align="center">
<img src="https://github.com/SarthakJShetty/Fracture-Detection/blob/master/Results/Sharpen.jpg"
alt="Sharpen Kerneling results"/>
</p>

- **Sobel X:** 

<p align="center">
<img src="https://github.com/SarthakJShetty/Fracture-Detection/blob/master/Results/Sobel%20X.jpg"
alt="Sobel X Kerneling results"/>
</p>

- **Sobel Y:** 

<p align="center">
<img src="https://github.com/SarthakJShetty/Fracture-Detection/blob/master/Results/Sobel%20Y.jpg"
alt="Sobel Y Kerneling results"/>
</p>