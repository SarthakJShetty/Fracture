# Fracture
Building a Computer Vision based tool for detecting fractures and fatiguing in mechanical components.

### :warning: Code is buggy :warning:

### Introduction:
- This project aims to develop a tool for identifying fractures and fissures in a mechanical component.

- The tool makes use of OpenCV and TensorFlow. OpenCV is used to visually detect the presence of the fracture and TensorFlow is used to predict the presence of fractures.

	 <strong>Note:</strong> Most of the TensorFlow code has been pulled from the TensorFlow repository sans a few changes.

### Overview of model:
<p align="center">
	<img src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Data/Pipeline_Overview_PNG.png" title="Outline of model">
	<figcaption><em>Fig 1. Block diagram of pipeline</em>
	</figcaption>
</p>

### Working:
1. A image is sent to the OpenCV code which runs it through a series of "Kernels", which include:

	- Sobel-X
	- Sobel-Y
	- Small Blur
	- Large Blur
	- Sharpen
	- Laplacian

2. The OpenCV code serves as a detector for fractures and relays it to the operator.

3. The image is then passed to the label_image.py [script](https://github.com/SarthakJShetty/Fracture/blob/master/label_image.py), which predicts whether the object is classified as fractured or not.

4. A retrain.py [script](https://github.com/SarthakJShetty/Fracture/blob/master/retrain.py) code is provided which is trained on the dataset of images. 

5. A webscraper has been developed which scrapes Google Images for the images to build your dataset (yet to be developed).

#### Usage:

1. Clone the repository:

	```git clone https://github.com/SarthakJShetty/Fracture.git```

2. Using the webscraper, scrape images from Google Images to build your dataset.

	<strong>Usage:</strong> ```python webscraper.py --search "Gears" --num_images 100 --directory /path/to/dataset/directory```

	<strong>Note:</strong> Make sure that both categories of images are in a common directory.

	<strong>Credits: This <a title="Webscraper" href="https://github.com/SarthakJShetty/Fracture/blob/master/webscraper.py">webscraper</a> was written by <a title="genekogan" href="http://genekogan.com/" target="_blank">genekogan</a>. All credits to him for developing the scrapper.</strong>

3. Retrain the final layers of Inception V3, to identify the images in the new dataset.

	<strong>Usage:</strong> ```python retrain.py --image_dir path/to/dataset/directory --path_to_files="project_name"```

	<strong>Note:</strong> The ```path_to_files``` creates a new file ```project_name``` under the ```tmp``` folder, and stores retrain logs, bottlenecks, checkpoints for the project here.</strong>

4. The previous step will cause logs and graphs to be generated during the training, and will take up a generous amount of space. We require the labels, bottlenecks and output graphs generated for the ```Labeller.py``` script.

5. We can now use ```Labeller.py``` to identify the whether the given component is defective or not. 

	<strong>Usage:</strong> ```python Labeller.py --graph=path/of/tmp/file/generated/output_graph.pb --labels=path/of/tmp/file/project_name/generated/labels.txt --output_layer=final_result```

6. The above step triggers the ```VideoCapture()``` function, which displays the camera feed. Once the specimen is in position, press the Q button on the keyboard, the script will retain the latest frame and pass it onto the ```Labeller.py``` and ```Kerneler.py``` programs.

#### Results of Kerneler:

- **Laplacian Kernel:** 
		<p align="center">
			<img title="Laplacian Filter" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Results/Laplacian_Gray.jpg"/>
			<figcaption>
				<em>Fig 2. Result of Laplacian Kernel</em>
			</figcaption>
		</p>

- **Sharpen:** 	
		<p align="center">
			<img title="Sharpening filter" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Results/Sharpen_Gray.jpg"/>
			<figcaption>
				<em>Fig 3. Result of Sharpening Kernel</em>
			</figcaption>
		</p>

- **Sobel X:** 
		<p align="center">
			<img title="Sobel-X filter" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Results/Sobel%20X_Gray.jpg"/>
			<figcaption>
			<em>Fig 4. Result of Sobel-X Kernel</em>
			</figcaption>
		</p>

- **Sobel Y:** 
		<p align="center">
			<img title="Sobel-Y filter" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Results/Sobel%20Y_Gray.jpg"/>
			<figcaption>
			<em>Fig 5. Result of Sobel-Y Kernel</em>
			</figcaption>
		</p>

#### Results of TensorFlow model:

- **<a title="Result 1" href="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Results/Predictions_Terminal_1.png">Prediction 1:</a>**
		<p align="center">
			<img title="Prediction 1" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Results/Predictions_Terminal_1.png">
			<figcaption>
			<em>Fig 6. Prediction 1 made by model</em>
			</figcaption>
		</p>

- **<a title="Result 2" href="">Prediction 2:</a>**
		<p align="center">
			<img title="Prediction 2" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Results/Predictions_Terminal_2.png">
			<figcaption>
			<em>Fig 7. Prediction 2 made by model</em>
			</figcaption>
		</p>

- **Train accuracy:**
		<p align="center">
			<img title="Training accuracy" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Data/TrainingAccuracy_vs_Steps.png">
			<figcaption>
			<em>Fig 8. Training Accuracy vs Steps</em>
			</figcaption>
		</p>

- **Validation accuracy:**
		<p align="center">
			<img title="Validation accuracy" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Data/ValidationAccuracy_vs_Steps.png">
			<figcaption>
			<em>Fig 9. Validation Accuracy vs Steps</em>
			</figcaption>
		</p>

- **Cross-entropy (Training):**
		<p align="center">
			<img title="Cross-entropy during training" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Data/TrainingEntropy_vs_Steps.png">
			<figcaption>
			<em>Fig 10. Training Entropy vs Steps</em>
			</figcaption>
		</p>

- **Cross-entropy (Validation):**
		<p align="center">
			<img title="Cross-entropy during validation" src="https://raw.githubusercontent.com/SarthakJShetty/Fracture/master/Data/ValidationEntropy_vs_Steps.png">
			<figcaption>
			<em>Fig 11. Validation Entropy vs Steps</em>
			</figcaption>
		</p>