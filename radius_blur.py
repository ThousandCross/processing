import numpy as np
import cv2

src = cv2.imread('tire2_rotatetest.jpg',0)

rows,cols  = src.shape[:2]
growMapx   = np.zeros(src.shape[:2],np.float32)
growMapy   = np.zeros(src.shape[:2],np.float32)
shrinkMapx = np.zeros(src.shape[:2],np.float32)
shrinkMapy = np.zeros(src.shape[:2],np.float32)

center_x = float(cols)/2
center_y = float(rows)/2
blur = 0.01
iterations = 8

for j in range(rows):
    for i in range(cols):
        growMapx.itemset((j,i),i + ((i - center_x) * blur))
        growMapy.itemset((j,i),j + ((j - center_y) * blur))
        shrinkMapx.itemset((j,i),i - ((i - center_x) * blur))
        shrinkMapy.itemset((j,i),j - ((j - center_y) * blur))

for k in range(iterations):
    tmp1 = cv2.remap(src, growMapx, growMapy, cv2.INTER_LINEAR)
    tmp2 = cv2.remap(src, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
    src  = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 2.2)

cv2.imwrite('radial_blur.jpg', src)
