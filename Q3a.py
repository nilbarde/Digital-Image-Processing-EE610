import cv2
import numpy as np
import math

def getDFT(img): # DFT function
	imgShape = img.shape
	imgResult = img.copy()
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			imgResult[i,j] = ((-1)**(i+j))*img[i,j]
	return np.fft.fft2(imgResult)

def getLogDFT(img): # DFT + log for getting DFT of image
	imgFFT = getDFT(img)
	imgFFT = np.abs(imgFFT)
	imgLogFFT = logTrans(imgFFT)

	return imgLogFFT

def IDFT(img): # IDFT function
	imgShape = img.shape
	imgResult = img.copy()
	imgIFFT = np.real(np.fft.ifft2(img))
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			imgResult[i,j] = ((-1)**(i+j))*imgIFFT[i,j]
	return imgResult.astype("uint8")

def logTrans(img):
	maxVal = np.amax(img) # max value in image
	cTarns = 255.0/np.log(1+maxVal) # scaling factor which sets max value to 255 

	imgTrans = cTarns*(np.log(img+1.001)) # log transformation
	imgTrans = imgTrans.astype("uint8") # converting back to image format

	return imgTrans

def smooth(img,kernel):
	imgShape = img.shape
	# padding image by zeros
	imgPad = cv2.copyMakeBorder(img,kernel.shape[0]//2,kernel.shape[0]//2,kernel.shape[1]//2,kernel.shape[1]//2,borderType=0)
	# padding kernel by zeros to make size as same as padded image 
	kernelPad = cv2.copyMakeBorder(kernel,0,imgShape[0]-1,0,imgShape[1]-1,borderType=0)

	# getting DFT of image and kernel
	kernelDFT = getDFT(kernelPad)
	imgDFT = getDFT(imgPad)

	# multiplying by kernel and retrieving result after IDFT
	resultFFT = imgPad*kernelPad
	result = IDFT(resultFFT)

	imgKernel = getLogDFT(kernelPad)
	cv2.imwrite('./outputs/q3a_filter.png', imgKernel)

	return result #returning the output


if __name__ == "__main__":
	kernel = np.ones((5,5))/(5*5)
	img = cv2.imread('./Data/q3a.png' , 0)
	r_im = getLogDFT(img)

	cv2.imwrite('./outputs/q3a_fft.png', r_im)

	result = smooth(img,kernel)
	cv2.imwrite('./outputs/q3a_out.png', result)
