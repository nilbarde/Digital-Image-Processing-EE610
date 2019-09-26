import cv2
import numpy as np

def globalHistTrans(img):
	hist, bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()

	cdfMasked = np.ma.masked_equal(cdf,0)

	cdfNorm = (cdfMasked - cdfMasked.min())*255.0/(cdfMasked.max()-cdfMasked.min())

	cdfResult = np.ma.filled(cdfNorm,0).astype("uint8")
	imgHist = cdfResult[img]

	return imgHist

def hist(img):
	probs = np.array([0.0 for i in range(256)])
	for x in img.flatten():
		probs[x] += 1

	probs /= np.sum(probs)

	s = img[img.shape[0]//2,img.shape[1]//2]
	imgMin, imgMax = img.min(), img.max()
	return imgMin + np.sum(probs[:s+1])*(imgMax-imgMin)

def localHistTrans(img,kernel):
	imgResult = img.copy()
	imgPad = cv2.copyMakeBorder(img,kernel.shape[0]//2,kernel.shape[0]//2,kernel.shape[1]//2,kernel.shape[1]//2,borderType=0)
	imgMin, imgMax = img.min(), img.max()
	imgShape = img.shape
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			imgResult[i][j] = round((hist(imgPad[i:i+kernel.shape[0],j:j+kernel.shape[1]])))
	return imgResult

if __name__ == "__main__":
	imgOriginal = cv2.imread("./Data/q2.png",0) # reading image using openCV
	x = 11
	kernel = np.ones((x,x))/(x*x)
	# imgGlobalHist = globalHistTrans(imgOriginal)
	imgLocalHist = localHistTrans(imgOriginal,kernel)

	# cv2.imshow("original",imgOriginal)
	# cv2.imshow("global histogram transformed",imgGlobalHist)
	# cv2.imshow("local histogram transformed",imgLocalHist)
	# cv2.imwrite("./outputs/q2_global histogram.png",imgGlobalHist)
	cv2.imwrite("./outputs/q2_local histogram_11x11.png",imgLocalHist)
	# cv2.waitKey(0)


