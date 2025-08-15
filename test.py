import cv2, numpy as np

src = cv2.imread("outputs/v9/results/type_4/F7HR_3176_R1.PNG", cv2.IMREAD_GRAYSCALE)
print("dtype:", src.dtype, "min/max/mean/std:", src.min(), src.max(), src.mean(), src.std())

# quick histogram
hist = cv2.calcHist([src],[0],None,[256],[0,256]).ravel()
print("nonzero bins:", np.count_nonzero(hist))
