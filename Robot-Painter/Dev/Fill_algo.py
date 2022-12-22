import numpy as np
import cv2
from skimage import segmentation, color
from skimage.segmentation import find_boundaries


MIN_CONTOUR_AREA = 5

img = cv2.imread('/home/leo/Downloads/Kozlova_art.jpeg')
final_labels = segmentation.slic(img, sigma=3, compactness=20, n_segments=1000, start_label=1, convert2lab=True, max_num_iter=100)
superpixels = color.label2rgb(final_labels, img, kind='avg')

result = np.zeros(img.shape, dtype=np.uint8)
cv2.imshow('img', superpixels)
cv2.waitKey(0)

def find_nearest_point(control_point, cnt):
    min = 10000000
    index = -1
    for i, point in enumerate(cnt):
        error = np.linalg.norm(control_point - point, 2)
        if min > error:
            index = i
            min = error

    if min >= 10:
        return -1

    return index

def fill(mask, border, color_, start_point, brush=3):

    # result =np.zeros(img.shape)

    if cv2.contourArea(border) < MIN_CONTOUR_AREA:
        return
    
    starting_index = find_nearest_point(start_point, border)

    border = np.roll(border, -starting_index, axis=0)

    for point in border:
        point = point[0]
        
        # cv2.circle(result, point, radius=1, color_=color_)
        for r in range(-brush+1, brush):
            try: 
                result[point[1]+r][point[0]] = (color_[0], color_[1], color_[2])
                mask[point[1] + r][point[0]] = 0
            except IndexError:
                print("BORDER WARNING")
            try: 
                result[point[1]][point[0]+r] = (color_[0], color_[1], color_[2])
                mask[point[1]][point[0] + r] = 0
            except IndexError:
                print("BORDER WARNING")
            try: 
                result[point[1]+r][point[0]+r] = (color_[0], color_[1], color_[2])
                mask[point[1] + r][point[0] + r] = 0
            except IndexError:
                print("BORDER WARNING")
        cv2.imshow('result', result)
        cv2.imshow('amsk', mask)

        cv2.waitKey(1)

            
    # mask = cv2.erode(mask, np.ones((4,4), np.uint8))
    # cv2.imshow('amsk', mask)
    # cv2.waitKey(0)
    # mask_cnt = np.zeros(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(f'[INFO]:\t Found {len(contours)} contours')
    # cv2.drawContours(mask_cnt, contours[0], -1, 255, thickness=-1)
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        print(f"[INFO]:\t Filling {i} contour")
        mask  = np.zeros(mask.shape, dtype=np.uint8)
        mask = cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        point = border[starting_index]
        fill(mask, cnt, color_, point, brush=brush)
        cv2.waitKey(1)

for i in np.unique(final_labels):
    mask = np.ones(img.shape) * (255, 255,255)
    mask[final_labels!=i] = (0,0,0)
    mask_color = img[np.where(final_labels==i)[0][1]][np.where(final_labels==i)[0][0]]
    print("+_____++_+_+_+_++++______+__+")
    print(mask_color)
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    border, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    border = border[0]
    M = cv2.moments(border)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    color_ = img[cy][cx]

    # print(border_paths[2][0][0] in mask[not np.array_equal(mask, (0,0,0))])

    # for j, border in border_paths.items():
    #     if not border:
    #         continue
    #     border = np.array([[np.array(border[i])] for i in range(len(border))])
    #     border = np.flip(border, axis=None)
    #     cv2.drawContours(img, [np.array(border)], -1, (0, 255,0), thickness=3)
    #     # M = cv2.moments(border)
    #     # cx = int(M['m10']/M['m00'])
    #     # cy = int(M['m01']/M['m00'])
    #     # cv2.putText(img, f'{j}', (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    # cv2.imshow('mask', img)
    # cv2.waitKey(0)

    fill(mask, border, color_, border[0][0])


cv2.waitKey(0)