import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def getFourier(img): # function for fourier
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)

	return fshift

def degradeFilter(shape): # filter defination
	filt = np.zeros((shape),dtype="complex_")
	for i in range(shape[0]):
		for j in range(shape[1]):
			u = i-shape[0]/2
			v = j-shape[1]/2
			real =  (0.001 + np.sin(np.pi*(0.1*(u) + 0.1*(v)))) * np.cos(np.pi*(0.1*(u) + 0.1*(v)))/(0.001 + np.pi*(0.1*(u) + 0.1*(v)))
			comp = -(0.001 + np.sin(np.pi*(0.1*(u) + 0.1*(v)))) * np.sin(np.pi*(0.1*(u) + 0.1*(v)))/(0.001 + np.pi*(0.1*(u) + 0.1*(v)))
			filt[i][j] = complex(real, comp)

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

	# degrades using given function
	ImgFiltered = applyFilter(imgFFT,FreqFilter)

	# cv2.imshow("Filtered",ImgFiltered)

	imgMax = img.max()
	StdDevi = [0.05*i*imgMax for i in range(6)]
	# making noise
	noise = [np.random.normal(0, StdDevi[i], ImgFiltered.shape) for i in range(6)]
	# fourier of noise
	noiseFFT = [getFourier(noise[i]) for i in range(len(noise))]
	# making noise added images
	noisyImgs = [ImgFiltered.astype(float) + noise[i] for i in range(6)]
	# fourier of noise added images
	noisyImgsFFT = [getFourier(noisyImgs[i]) for i in range(len(noisyImgs))]

	# calc Nff and Pff
	Nff = [np.conj(i)*i for i in noiseFFT]
	Pff = np.conj(imgFFT)*imgFFT
	# making wiener filter
	FreqFilterVal = np.conj(FreqFilter)*(FreqFilter)
	WienerFilter = [FreqFilterVal/(FreqFilterVal+(Nff[i]/Pff)) for i in range(len(Nff))]

	# inverse filtering on noise images
	noisyImgsFiltered = [applyFilter(noisyImgsFFT[i],WienerFilter[i]) for i in range(len(noisyImgsFFT))]

	for i in range(len(noisyImgs)):
		cv2.imshow(str(i),noisyImgsFiltered[i])
		cv2.imwrite("./Results/Q2.2 std devi : " + str(round(StdDevi[i],2)) + " filtered output.png",noisyImgsFiltered[i])

	# RRMSE calc
	RRMSEnoisy =[(np.sum((img - noisyImgs[i])**2)*1.0/np.sum((img)**2))**0.5 for i in range(len(noisyImgs))]
	RRMSEfiltered =[(np.sum((img - noisyImgsFiltered[i])**2)*1.0/np.sum((img)**2))**0.5 for i in range(len(noisyImgs))]

	plt.plot(StdDevi,RRMSEnoisy)
	plt.savefig("./Results/Q2.2.1 - RRMSE degraded image.png")
	plt.show()
	plt.plot(StdDevi,RRMSEfiltered)
	plt.savefig("./Results/Q2.2.2 - RRMSE filtered images.png")
	plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()
