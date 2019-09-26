import cv2
import numpy as np

def logTrans(img):
	maxVal = np.amax(img) # max value in image
	cTarns = 255.0/np.log(1+maxVal) # scaling factor which sets max value to 255 

	imgTrans = cTarns*(np.log(img+1.001)) # log transformation
	imgTrans = imgTrans.astype("uint8") # converting back to image format

	return imgTrans

def antilogTrans(img):
	maxVal = np.amax(img) # max value in image
	imgTemp = (np.exp(img*1.0/maxVal)-1) # taking antilog (range from 0 to e-1 )

	imgTrans = 255*(imgTemp/(np.e-1)) # scaling max value to 255
	imgTrans = imgTrans.astype("uint8") # converting back to image format

	return imgTrans

if __name__ == "__main__":
	imgOriginal = cv2.imread("./Data/q1.png",0) # reading image using openCV

	imgLog = logTrans(imgOriginal) # log transformation function
	imgAntilog = antilogTrans(imgOriginal) # antilog transformation function

	cv2.imwrite("./outputs/q1_log.png",imgLog)
	cv2.imwrite("./outputs/q1_antilog.png",imgAntilog)
