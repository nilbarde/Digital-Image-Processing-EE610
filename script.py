import cv2
import numpy as np

def get_dark_channel(I, w):
	M, N, _ = I.shape
	padded = np.pad(I, ((w / 2, w / 2), (w / 2, w / 2), (0, 0)), 'edge')
	darkch = np.zeros((M, N))
	for i, j in np.ndindex(darkch.shape):
		darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
	return darkch

def get_gamma_corrected(img,gamma):
	img = img.copy().astype(float)
	img /= 255.0
	img = img**gamma
	img *= 255.0
	return img.astype("uint8")

imgName = "./images/f1.jpg"
imgNameWE = imgName[:-4]
img = cv2.imread(imgName)

# cv2.imwrite(imgNameWE + "-0-original.png",img)

dark = get_dark_channel(img,7)
darkName = imgNameWE + "-1-dark.png"
cv2.imwrite(darkName,dark)

tx = (1 - 0.9*dark/(255.0))*255
tx = tx.astype("uint8")
txName = imgNameWE + "-2-tx.png"
cv2.imwrite(txName,tx)

jt = img.copy()
for c in range(3):
	jt[:,:,c] = img[:,:,c] - 180*(1-(tx/255.0))
jtName = imgNameWE + "-3-jt.png"
cv2.imwrite(jtName,jt)

j = jt.copy()
for c in range(3):
	j[:,:,c] = (jt[:,:,c]*255.0)/(tx)
jName = imgNameWE + "-4-j.png"
cv2.imwrite(jName,j)

imgGamma = get_gamma_corrected(j,1.2)
gammaName = imgNameWE + "-5-gamma.png"
cv2.imwrite(gammaName,imgGamma)

