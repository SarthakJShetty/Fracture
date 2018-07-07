from skimage.exposure import rescale_intensity
import numpy as np
from skimage.measure import compare_ssim
import argparse
import cv2
import imutils
import scipy as sp
import datetime

#Convolution function, takes in the image and the kernel for processing, applies the required filters required
def convolve(image, kernel):
	(iH,iW)=image.shape[:2]
	(kH,kW)=kernel.shape[:2]
	pad=(kW-1)/2
	pad=int(pad)
	image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
	output=np.zeros((iH,iW),dtype="float32")

	for y in np.arange(pad,iH+pad):
		for x in np.arange(pad,iW+pad):
			roi=image[y-pad:y+pad+1,x-pad:x+pad+1]
			k=(roi*kernel).sum()
			output[y-pad,x-pad]=k
	output=rescale_intensity(output,in_range=(0,255))
	output=(output*255).astype("uint8")
	return output

def run_kerneler(image):
	#This is an experimental section added to check whether the color_mode variable was being received by Kerneler.py or not
	'''if (int(color_mode)==0):
		print("[INFO] Color mode has been received and the color mode is: "+color_mode)
	print("[INFO]You have entered the Kerneler\n")'''

	#Standard argument parser intro'd here, takes into consideration two images, that have to be compared
	'''ap=argparse.ArgumentParser()
	ap.add_argument("-i","--Image_Insert",required=True,help="Add the path to the first image")
	args=vars(ap.parse_args())'''

#Kernels start from here
#Blurring kernels
	#smallBlur=np.ones((7,7),dtype="float")*(1.0/(7.0*7.0))
	#largeBlur=np.ones((21,21),dtype="float")*(1.0/(21*21))

#Sharpening Kernels
	sharpen=np.array((
	[0,-1,0],
	[-1,5,-1],
	[0,-1,0]),dtype="int")

#Laplacian Kerner
	laplacian=np.array((
	[0,1,0],
	[1,-4,1],
	[0,1,0]),dtype="int")

#Sobel Kernel in the X direction, detects the edges only in the X direction
	sobelX=np.array((
	[-1,0,1],
	[-2,0,2],
	[-1,0,1]),dtype="int")

#Sobel Kernel in the Y direction, detects the edges only in the Y direction
	sobelY=np.array((
	[-1,-2,-1],
	[0,0,0],
	[1,2,1]),dtype="int")

#Kernel bank, sends the image and the kernel to convolve function, to convolute it
	kernelBank=(
	#("Small Blurring",smallBlur),
	#("Large Blurring",largeBlur),
	("Sharpen",sharpen),
	("Sobel X",sobelX),
	("Sobel Y",sobelY),
	("Laplacian",laplacian))
	image=cv2.imread(image)
	blur=cv2.GaussianBlur(image,(3,3),0)
	blur=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blur=cv2.resize(blur,(0,0),fx=.5,fy=.5)
	gray=cv2.resize(gray,(0,0),fx=.5,fy=.5)
	cv2.imwrite('Results/Image_Analyzed.jpg',image)
	stored_image=cv2.copyMakeBorder(blur,0,50,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
	stored_image=cv2.putText(stored_image,"Original Image",(10,(stored_image.shape[0]-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,100),1,cv2.LINE_AA)
	stored_image=cv2.rectangle(stored_image,(0,(stored_image.shape[0]-50)),(stored_image.shape[1],stored_image.shape[0]),(0,0,0),3)
	cv2.imwrite('Results/Stored_Image.jpg',stored_image)
	cv2.imshow("Stored_Image",stored_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#Adding first image here
#image_2=cv2.resize(image_2,(0,0),fx=.5,fy=.5)

#Converting image to grayscale here
#Displaying images here
#Cycles through all the kernels available in the bank, compares with the image
	for(kernelName,kernel) in kernelBank:
		print("[INFO]Applying {} kernel".format(kernelName))
		convolve_gray_Output=convolve(gray,kernel)
		convolve_blur_Output=convolve(blur,kernel)
		convolve_gray_Output=cv2.copyMakeBorder(convolve_gray_Output,0,50,0,0,cv2.BORDER_CONSTANT,value=[255,0,0])
		convolve_gray_Output=cv2.putText(convolve_gray_Output,"Filter: "+kernelName+" "+"Mode: Gray",(10,(convolve_gray_Output.shape[0]-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,100),1,cv2.LINE_AA)
		convolve_gray_Output=cv2.rectangle(convolve_gray_Output,(0,(convolve_gray_Output.shape[0]-50)),(convolve_gray_Output.shape[1],convolve_gray_Output.shape[0]),(0,0,0),3)
		convolve_blur_Output=cv2.copyMakeBorder(convolve_blur_Output,0,50,0,0,cv2.BORDER_CONSTANT,value=[255,0,0])
		convolve_blur_Output=cv2.putText(convolve_blur_Output,"Filter: "+kernelName+" "+"Mode: Color",(10,(convolve_blur_Output.shape[0]-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,100,100),1,cv2.LINE_AA)
		convolve_blur_Output=cv2.rectangle(convolve_blur_Output,(0,(convolve_blur_Output.shape[0]-50)),(convolve_blur_Output.shape[1],convolve_blur_Output.shape[0]),(0,0,0),3)
		cv2.imwrite('Results/'+kernelName+"_Gray"+".jpg", convolve_gray_Output)
		cv2.imwrite('Results/'+kernelName+"_Color"+".jpg", convolve_blur_Output)
		#opencvOutput=cv2.filter2D(gray,-1,kernel)
		#No need to call difference image here
		#(score)=compare_ssim(convolveOutput,opencvOutput,full=True)
		#print("[INFO]Score for {} kernel is: {:.4f}\n".format(kernelName,score[0]))
		#cv2.imshow("original",gray)
		cv2.imshow("{}-gray".format(kernelName),convolve_gray_Output)
		cv2.imshow("{}-color".format(kernelName),convolve_blur_Output)
		#cv2.imshow("{}-opencv".format(kernelName),opencvOutput)
		cv2.waitKey(0)
		cv2.destroyAllWindows()