import numpy as np
import cv2

src  = cv2.imread('tire1.jpg', 1)
rows, cols, channels  = src.shape[:3]
size = tuple(np.array([cols, rows]))

blur = 10
iterations = 8

# rotate
rad = 0
# 
move_x = 100
# 
move_y = 0

matrix = [
            [np.cos(rad),  -1 * np.sin(rad), move_x],
            [np.sin(rad),   np.cos(rad), move_y]
         ]

affine_matrix = np.float32(matrix)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img_maskg = cv2.threshold(src_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
ret, contours, hierarchy = cv2.findContours(img_maskg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.fillPoly(img_maskg, contours, color=(255,255,255))
img_mask = cv2.merge((img_maskg, img_maskg, img_maskg))
img_src2m = cv2.bitwise_and(src, img_mask)
img_maskn = cv2.bitwise_not(img_mask)
tmp_src = src

for k in range(iterations):
    # create mask of next src
    src_next  = cv2.warpAffine(src, affine_matrix, size, cv2.INTER_LINEAR)
    src_next[: , 0:(move_x * (k + 1))] = [255,255,255]

    # create img_mask_part1
    tmp_src_next  = cv2.warpAffine(tmp_src, affine_matrix, size, cv2.INTER_LINEAR)
    tmp_src_next[: , 0:(move_x * (k + 1))] = [255,255,255]
    tmp_img_next_gray = cv2.cvtColor(tmp_src_next, cv2.COLOR_BGR2GRAY)
    tmp_img_next_maskg = cv2.threshold(tmp_img_next_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    ret3, contours3, hierarchy3 = cv2.findContours(tmp_img_next_maskg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(tmp_img_next_maskg, contours3, color=(255,255,255))
    tmp_img_next_mask = cv2.merge((tmp_img_next_maskg, tmp_img_next_maskg, tmp_img_next_maskg))
    img_mask_part1 = cv2.bitwise_and(img_maskn, tmp_img_next_mask) 

    # create img_mask_part2
    img_mask_part2_pre = cv2.bitwise_or(img_mask, tmp_img_next_mask)
    img_mask_part2 = cv2.bitwise_not(img_mask_part2_pre)
    
    # alpha blending
    w1 = 0.05 * (k + 1)
    w2 = 1 - 0.05 * (k + 1) 
    tmp1 = cv2.addWeighted(src, w1, src_next, w2, 2.2)
    tmp1 = cv2.bitwise_and(tmp1, img_mask_part1) 
    tmp2 = cv2.bitwise_and(src, img_mask_part2) 
    tmp3 = cv2.bitwise_and(src, img_mask)

    output = cv2.bitwise_or(tmp1, tmp2)
    output = cv2.bitwise_or(output, tmp3)
    cv2.imwrite('output' + str(k + 1) + '.jpg', output)
    
    # prepare for next repeat
    src = output
    img_mask = img_mask_part2_pre
    tmp_src = tmp_src_next
    img_maskn = cv2.bitwise_not(tmp_img_next_mask)
