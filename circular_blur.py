import numpy as np
import cv2
import math

src = cv2.imread('tire3.jpg',0)

rows,cols  = src.shape[:2]
growMapx   = np.zeros(src.shape[:2],np.float32)
growMapy   = np.zeros(src.shape[:2],np.float32)
shrinkMapx = np.zeros(src.shape[:2],np.float32)
shrinkMapy = np.zeros(src.shape[:2],np.float32)

center_x = float(cols)/2
center_y = float(rows)/2
angle = 5
iterations = 10
rd = math.radians(angle)
rd2 = (-1) * rd

for j in range(rows):
    for i in range(cols):
        x = i - center_x
        y = j - center_y
        growMapx.itemset((j,i), x * math.cos(rd) - y * math.sin(rd) + center_x)
        growMapy.itemset((j,i), x * math.sin(rd) + y * math.cos(rd) + center_y)
        shrinkMapx.itemset((j,i), x * math.cos(rd2) - y * math.sin(rd2) + center_x)
        shrinkMapy.itemset((j,i), x * math.sin(rd2) + y * math.cos(rd2) + center_y)

for k in range(iterations):
    tmp1 = cv2.remap(src, growMapx, growMapy, cv2.INTER_LINEAR)
    tmp2 = cv2.remap(src, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
    src  = cv2.addWeighted(tmp1, 0.5, src, 0.5, 2.2)

# cv2.imwrite('circular_tire3_5_10.jpg', tmp1)
# cv2.imwrite('circular_tire3_5_10.jpg', tmp2)
cv2.imwrite('circular_tire3_5_10.jpg', src)
