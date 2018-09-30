import numpy as np
import sys
import cv2
import math

src = cv2.imread('tire1.png', 1)
rows, cols, channels = src.shape[:3]

growMapx   = np.zeros((rows,cols), np.float32)
growMapy   = np.zeros((rows,cols), np.float32)
shrinkMapx = np.zeros((rows,cols), np.float32)
shrinkMapy = np.zeros((rows,cols), np.float32)

center_x = float(cols - 1)/2
center_y = float(rows - 1)/2
# angle = 2
iterations = 8
blur = 0.01
#rd = math.radians(angle)
#rd2 = (-1) * rd

for j in range(rows):
    for i in range(cols):
        x = i - center_x
        y = j - center_y
        r = math.sqrt(x*x + y*y)
        rd = math.pi / 256 * r
        rd2 = (-1) * rd
        growMapx.itemset((j,i), x * math.cos(rd) - y * math.sin(rd) + center_x)
        growMapy.itemset((j,i), x * math.sin(rd) + y * math.cos(rd) + center_y)
        shrinkMapx.itemset((j,i), x * math.cos(rd2) - y * math.sin(rd2) + center_x)
        shrinkMapy.itemset((j,i), x * math.sin(rd2) + y * math.cos(rd2) + center_y)

for k in range(iterations):
    tmp1 = cv2.remap(src, growMapx, growMapy, cv2.INTER_LINEAR)
    tmp2 = cv2.remap(src, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
    src  = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0.1)

cv2.imwrite('output_swirl_tmp1.jpg', tmp1)
cv2.imwrite('output_swirl_tmp2.jpg', tmp2)
cv2.imwrite('output_swirl.jpg', src)

