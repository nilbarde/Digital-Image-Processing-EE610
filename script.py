import cv2
import numpy as np

def get_dark_channel(I, w):
	M, N, _ = I.shape
	padded = np.pad(I, ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2)), (0, 0)), 'edge')
	darkch = np.zeros((M, N))
	for i, j in np.ndindex(darkch.shape):
		darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
	darkch = darkch.astype("uint8")
	return cv2.cvtColor(darkch,cv2.COLOR_GRAY2RGB)

def get_gamma_corrected(img,gamma):
	img = img.copy().astype(float)
	img /= 255.0
	img = img**gamma
	img *= 255.0
	return img.astype("uint8")

imgName = "./images/f2.jpg"
imgNameWE = imgName[:-4]
img = cv2.imread(imgName)

# cv2.imwrite(imgNameWE + "-0-original.png",img)

dark = get_dark_channel(img,7)
darkName = imgNameWE + "-1-dark.png"
cv2.imwrite(darkName,dark)

tx = (1 - 0.9*dark/(255.0))*255
txName = imgNameWE + "-2-tx.png"
cv2.imwrite(txName,tx.astype("uint8"))

jt = img - 180*(1-(tx/255.0))
jtName = imgNameWE + "-3-jt.png"
cv2.imwrite(jtName,jt)

j = jt/(tx/255.0)
jName = imgNameWE + "-4-j.png"
cv2.imwrite(jName,j)

j = j.clip(0,255)

imgGamma = get_gamma_corrected(j,1.5)
gammaName = imgNameWE + "-5-gamma.png"
cv2.imwrite(gammaName,imgGamma)
