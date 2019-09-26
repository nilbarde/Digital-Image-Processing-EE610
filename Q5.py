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

def BLPF(img,d,n): #butterworth low pass filter
	imgShape = img.shape
	mask = np.zeros(imgShape)
	# calculating mask using BLPF function
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			mask[i][j] = 1/((1+(((i-imgShape[0]/2.0)**2+(j-imgShape[1]/2.0)**2)/(d*d)))**n)
	cv2.imwrite("./outputs/q5_b.png",logTrans(mask))
	# getting DFT of image and multiplying by mask
	imgDFT = getDFT(img)
	resultDFT = np.multiply(mask,imgDFT)
	# getting result by applying IDFT on DFT-result 
	result = IDFT(resultDFT)

	return result

def GLPF(img,d): # gaussian low pass filter
	imgShape = img.shape
	mask = np.zeros(imgShape)
	# calculating mask using GLPF function
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			mask[i][j] = math.exp(((i-imgShape[0]/2.0)**2+(j-imgShape[1]/2.0)**2)/(-2.0*d*d))
	cv2.imwrite("./outputs/q5_g.png",logTrans(mask))
	# getting DFT of image and multiplying by mask
	imgDFT = getDFT(img)
	resultDFT = np.multiply(mask,imgDFT)
	# getting result by applying IDFT on DFT-result 
	result = IDFT(resultDFT)

	return result

def ILPF(img,d): # ideal low pass filter
	imgShape = img.shape
	mask = np.zeros(imgShape)
	# calculating mask using ILPF function
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			if((((i-imgShape[0]/2.0)**2+(j-imgShape[1]/2.0)**2)**0.5)<d):
				mask[i][j] = 1
	cv2.imwrite("./outputs/q5_i.png",logTrans(mask))
	# getting DFT of image and multiplying by mask
	imgDFT = getDFT(img)
	resultDFT = np.multiply(mask,imgDFT)
	# getting result by applying IDFT on DFT-result 
	result = IDFT(resultDFT)

	return result

if __name__ == "__main__":
	imgOriginal = cv2.imread("./Data/q5.png",0) # reading image using openCV
	imgILPF = ILPF(imgOriginal,50)
	imgGLPF = GLPF(imgOriginal,50)
	imgBLPF = BLPF(imgOriginal,50,2)

	cv2.imwrite("./outputs/q5_ILPF.png",imgILPF)
	cv2.imwrite("./outputs/q5_GLPF.png",imgGLPF)
	cv2.imwrite("./outputs/q5_BLPF.png",imgBLPF)

