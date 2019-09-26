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

def laplacianFourierTrans(img,kernel):
	imgResult = img.copy()

	imgShape = img.shape
	imgPad = cv2.copyMakeBorder(img,kernel.shape[0]//2,kernel.shape[0]//2,kernel.shape[1]//2,kernel.shape[1]//2,borderType=0)
	kernelPad = cv2.copyMakeBorder(kernel,0,imgShape[0]-1,0,imgShape[1]-1,borderType=0)

	kernelDFT = getDFT(kernelPad)
	imgDFT = getDFT(imgPad)

	resultFFT = imgPad*kernelPad
	result = IDFT(resultFFT)

	imgKernel = getLogDFT(kernelPad)
	cv2.imwrite('./outputs/q4_filter.png', imgKernel)

	return result #returning the output

def laplacianTrans(img,kernel):
	imgResult = img.copy()
	imgPad = cv2.copyMakeBorder(img,1,1,1,1,borderType=0)
	imgShape = img.shape
	imgResult[:,:] = imgResult[:,:] + 1*((4*imgPad[1:-1,1:-1]) - (1*imgPad[:-2,1:-1]) - (1*imgPad[2:,1:-1]) - (1*imgPad[1:-1,:-2]) - (1*imgPad[1:-1,2:]))
	imgResult = np.clip(imgResult,0,255).astype("uint8")
	return imgResult

if __name__ == "__main__":
	imgOriginal = cv2.imread("./Data/q4.png",0) # reading image using openCV
	
	kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	imgLaplacian = laplacianTrans(imgOriginal,kernel)
	imgFourierLaplacian = laplacianFourierTrans(imgOriginal,kernel)

	cv2.imwrite("./outputs/q4_spatial.png",imgLaplacian)
	cv2.imwrite("./outputs/q4_fourier.png",imgFourierLaplacian)


