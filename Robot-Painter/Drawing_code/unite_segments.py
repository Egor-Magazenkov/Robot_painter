import numpy as np
import cv2
import scipy.spatial as sp
import signal
import sys

COLORS = np.loadtxt('colors.txt', np.uint0)
COLORS_ORDER_IDX = list(range(len(COLORS)))

def find_nearest_color(color):
    global COLORS
    diff = np.linalg.norm(np.array(color) - COLORS, axis=1)
    return np.argmin(diff)


if __name__ == '__main__':
    img_with_branch = cv2.imread('./branch.jpg')
    segment_image = cv2.imread('./mask.jpg')
    im = img_with_branch.astype(np.uint8)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, cv2.CV_64F)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
    
    img_branch_united = np.ones(img.shape)*255
   
    for c in range(COLORS.shape[0]):
        img = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
        for cnt in contours:
            clr = segment_image[cnt[0][0][1]][cnt[0][0][0]]
            clr = (int (clr[0]), int (clr[1]), int (clr[2]))
            clr_idx = find_nearest_color(clr)
            if (clr_idx == c):
                color = COLORS[clr_idx]
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                cv2.fillPoly(img, pts=[cnt], color=color)
                cv2.drawContours(img, [cnt], -1, color, 3)
                
        img = img.astype(np.uint8)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, cv2.CV_64F)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours2, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_branch_united, contours2 , -1, (0,0,0), 2)
    cv2.imshow("result", img_branch_united)    
    cv2.imwrite("branch_united.jpg", img_branch_united)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
