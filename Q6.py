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

def BHPF(img,d,n): # butterworth high pass filter
	imgShape = img.shape
	mask = np.zeros(imgShape)
	# calculating mask using BHPF function
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			xxx = ((i-imgShape[0]/2.0)**2+(j-imgShape[1]/2.0)**2)
			if xxx:
				mask[i][j] = 1/((1+((d*d)/xxx))**n)
	cv2.imwrite("./outputs/q6_b.png",logTrans(mask))
	# getting DFT of image and multiplying by mask
	imgDFT = getDFT(img)
	resultDFT = np.multiply(mask,imgDFT)
	# getting result by applying IDFT on DFT-result 
	result = IDFT(resultDFT)

	return result

def GHPF(img,d): # gaussian high pass filter
	imgShape = img.shape
	mask = np.zeros(imgShape)
	# calculating mask using GHPF function
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			mask[i][j] = 1.0 - math.exp(((i-imgShape[0]/2.0)**2+(j-imgShape[1]/2.0)**2)/(-2.0*d*d))
	cv2.imwrite("./outputs/q6_g.png",logTrans(mask))
	# getting DFT of image and multiplying by mask
	imgDFT = getDFT(img)
	resultDFT = np.multiply(mask,imgDFT)
	# getting result by applying IDFT on DFT-result 
	result = IDFT(resultDFT)

	return result

def IHPF(img,d): # ideal high pass filter
	imgShape = img.shape
	mask = np.zeros(imgShape)
	# calculating mask using IHPF function
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			if((((i-imgShape[0]/2.0)**2+(j-imgShape[1]/2.0)**2)**0.5)>d):
				mask[i][j] = 1
	cv2.imwrite("./outputs/q6_i.png",logTrans(mask))
	# getting DFT of image and multiplying by mask
	imgDFT = getDFT(img)
	resultDFT = np.multiply(mask,imgDFT)
	# getting result by applying IDFT on DFT-result 
	result = IDFT(resultDFT)

	return result

if __name__ == "__main__":
	imgOriginal = cv2.imread("./Data/q6.png",0) # reading image using openCV
	imgIHPF = IHPF(imgOriginal,10)
	imgGHPF = GHPF(imgOriginal,10)
	imgBHPF = BHPF(imgOriginal,10,2)

	cv2.imwrite("./outputs/q6_IHPF.png",imgIHPF)
	cv2.imwrite("./outputs/q6_GHPF.png",imgGHPF)
	cv2.imwrite("./outputs/q6_BHPF.png",imgBHPF)

