import numpy as np



def cnnPreprocess(img):
	"""
	This function will make the input image into the same format as the input parameters required for CNN Classification.

	Params:
		img: The image to be converted
	Returns:
		img: The image in the same size and format required by the CNN.

	"""
	#Set image type to float32
	img = img.astype('float32')
	#divide the image by 255
	img /= 255
	#Expand the dimensions of the image using numpy
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	#Return the resized image to the main function.
	return img


