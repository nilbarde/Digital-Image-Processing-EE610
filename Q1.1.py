import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def getFourier(img): # function for fourier
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)

	return fshift

def degradeFilter(shape): # filter defination
	filt = np.zeros((shape),dtype="float")
	for i in range(shape[0]):
		for j in range(shape[1]):
			filt[i][j] = (np.e)**(-0.0025*(((i-shape[0]/2)**2 + (j-shape[1]/2)**2)**(5.0/6)))
	return filt

def applyFilter(imgFFT,FreqFilter): # function which applies filter on image (both in fourier domain) and returns normal image
	FreqFiltered = imgFFT*FreqFilter
	FreqFilteredShift = np.fft.fftshift(FreqFiltered)
	Filtered = np.fft.ifft2(FreqFilteredShift)
	ImgFiltered = np.abs(Filtered)
	ImgFiltered = ((ImgFiltered - ImgFiltered.min())*255.0)/(ImgFiltered.max() - ImgFiltered.min())
	ImgFiltered = ImgFiltered.astype("uint8")

	return ImgFiltered

if __name__ == "__main__":
	# opening image in single channel mode
	img = cv2.imread("image.png",0)
	# cv2.imshow("img",img)

	# converts in fourier domain
	imgFFT = getFourier(img)
	# making filter
	FreqFilter = degradeFilter(imgFFT.shape)
	# inverse filter
	invFreqFilter = 1.0/(FreqFilter + 0.1)

	# degrades using given function
	ImgFiltered = applyFilter(imgFFT,FreqFilter)

	cv2.imshow("Filtered",ImgFiltered)

	imgMax = img.max()
	StdDevi = [0.05*i*imgMax for i in range(6)]
	# adding noise
	noisyImgs = [ImgFiltered.astype(float) + np.random.normal(0, StdDevi[i], ImgFiltered.shape) for i in range(6)]
	# gaussian conversion
	noisyImgs = [gaussian_filter(noisyImgs[i],sigma=i) for i in range(len(noisyImgs))]
	# fourier of noise added images
	noisyImgsFFT = [getFourier(noisyImgs[i]) for i in range(len(noisyImgs))]
	# inverse filtering on noise images
	noisyImgsFiltered = [applyFilter(i,invFreqFilter) for i in noisyImgsFFT]

	for i in range(len(noisyImgs)):
		cv2.imshow(str(i),noisyImgsFiltered[i])
		cv2.imwrite("./Results/Q1.1 std devi : " + str(round(StdDevi[i],2)) + " filtered output.png",noisyImgsFiltered[i])

	# RRMSE calculations
	RRMSEnoisy =[(np.sum((img - noisyImgs[i])**2)*1.0/np.sum((img)**2))**0.5 for i in range(len(noisyImgs))]
	RRMSEfiltered =[(np.sum((img - noisyImgsFiltered[i])**2)*1.0/np.sum((img)**2))**0.5 for i in range(len(noisyImgs))]

	plt.plot(StdDevi,RRMSEnoisy)
	plt.savefig("./Results/Q1.1.1 - RRMSE degraded image.png")
	plt.show()
	plt.plot(StdDevi,RRMSEfiltered)
	plt.savefig("./Results/Q1.1.2 - RRMSE filtered images.png")
	plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()
