import cv2
import numpy as np

def spatialSmoothTrans(img,kernel):
	imgResult = img.copy()
	# padding by zeros
	imgPad = cv2.copyMakeBorder(img,kernel.shape[0]//2,kernel.shape[0]//2,kernel.shape[1]//2,kernel.shape[1]//2,borderType=0)
	imgShape = img.shape
	# multiplying by kernel
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			imgResult[i][j] = np.sum(imgPad[i:i+kernel.shape[0],j:j+kernel.shape[1]]*kernel)
	return imgResult

def medianTrans(img,kernel):
	imgResult = img.copy()
	# zero padding
	imgPad = cv2.copyMakeBorder(img,kernel.shape[0]//2,kernel.shape[0]//2,kernel.shape[1]//2,kernel.shape[1]//2,borderType=0)
	imgShape = img.shape
	# getting median for each image kernel
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			imgResult[i][j] = np.median(imgPad[i:i+kernel.shape[0],j:j+kernel.shape[1]])
	return imgResult

if __name__ == "__main__":
	imgOriginal = cv2.imread("./Data/q3b.png",0) # reading image using openCV
	kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]).astype(float)
	kernel /= np.sum(kernel)
	imgSmooth = spatialSmoothTrans(imgOriginal,kernel)

	kernel = np.ones((3,3))/(3*3)
	imgMedian = medianTrans(imgOriginal,kernel)

	cv2.imwrite("./outputs/q3b_smooth.png",imgSmooth)
	cv2.imwrite("./outputs/q3b_median.png",imgMedian)

