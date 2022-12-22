import cv2
import numpy as np
import sys
import os
import roboticstoolbox.tools.trajectory as rtb
import pickle

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



img = cv2.imread('./branch.jpg')
source_image = cv2.imread('./mask.jpg', cv2.IMREAD_UNCHANGED)


kernel = np.ones((5, 5), 'uint8')
img_contours_global = np.zeros(img.shape, np.uint8)
filled_contour = np.array([[]])
binary_mask = np.zeros(img.shape[0:2])

COLORS = np.loadtxt('colors.txt', np.uint0)
COLORS_ORDER_IDX = list(range(len(COLORS)))

data = {
    'trajectories':[]
}

count = 0
PROHIBITED_AREA = 1

def reshape_contour(cnt):
    trajectory = cnt
    trajectory = np.reshape(trajectory, (len(trajectory), -1))
    trajectory = np.array(trajectory)
    trajectory = trajectory.astype(np.float64)

    return trajectory

def check_area(cnt):
    global PROHIBITED_AREA

    return 	cv2.contourArea(cnt) > PROHIBITED_AREA

def save_to_file(cnt):
    global count
    global current_color
    global COLORS
    global data
    global source_image

    x = int(cnt[0][0])
    y = int(cnt[0][1])

    current_color = source_image[y][x]
    current_color = [int (current_color[0]), int (current_color[1]), int (current_color[2])]

    label_color = find_nearest_color(current_color)



    trajectory = cnt.copy()
    trajectory[:, 1] = 400-trajectory[:, 1]
    trajectory /= 1000.0

    trj_array = rtb.mstraj(trajectory, dt=0.002, qdmax=0.25, tacc=0.05)
    trj = {'points': trj_array.q, 'width': 1.0, 'color': label_color}
    data['trajectories'].append(trj)

def get_contour(img, flag): # if flag is 0 will return all contours else will return internal contours
    #convert img to grey
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
    #set a thresh
    thresh = 10
    #get threshold image
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if flag == 0:
        return contours

    if len(contours) == 0:
        return []

    contour_index = hierarchy[0]
    for i, index in enumerate(contour_index):
        cnt = contours[i]
        # fill_contour(cnt, img)

def get_color(cnt):
    global source_image
    global current_color

    x = cnt[0][0][0]
    y = cnt[0][0][1]

    current_color = source_image[y][x]
    color = (int (current_color[0]), int (current_color[1]), int (current_color[2]))
    # color = list(np.random.random(size=3) * 256)
    return color


def find_nearest_color(color):
    global COLORS

    diff = np.linalg.norm(np.array(color) - COLORS, axis=1)
    # print(diff[np.argmin(diff)])
    return np.argmin(diff)


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

def fill_contour(cnt, IMG):
    global img_contours_global
    global filled_contour
    global COLORS

    if check_area(cnt) == False:
        return

    color = COLORS[find_nearest_color(get_color(cnt))]
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.drawContours(img_contours_global, [cnt], 0, color, 3)

    cnt = reshape_contour(cnt)

    if filled_contour.shape[1] != 0:
        last_point = filled_contour[-1]
        index = find_nearest_point(last_point, cnt)
        if index == -1:
            save_to_file(filled_contour)
            filled_contour = cnt
        else:
            cnt = np.roll(cnt, -index, axis=0)
            filled_contour = np.concatenate((filled_contour, cnt), axis=0)
    else:
        filled_contour = cnt

    cv2.imshow('parent contour', img_contours_global)
    cv2.waitKey(2)

    # kernel = np.ones((3, 3 ), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(IMG, kernel, iterations = 1)

    contours = get_contour(erosion, 1)

    return


# Main part
if __name__ == '__main__':

    current_color = [0, 0, 0]

    contours = get_contour(img, 0)
    img_with_contour = np.zeros(img.shape, np.uint8)

    for i in range(len(contours)):
        filled_contour = np.array([[]])
        if (len(contours[i]) <= 1) or cv2.contourArea(contours[i]) < 100:
            continue

        color = COLORS[find_nearest_color(get_color(contours[i]))]
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.drawContours(img_with_contour, [contours[i]], 0, color, thickness=cv2.FILLED)
        cv2.imshow('filled_img', img_with_contour)
        kernel = np.ones((8, 8), np.uint8)
        dilation = cv2.dilate(img_with_contour, kernel, iterations = 1)


        fill_contour(contours[i], dilation)
        if (len(filled_contour) <= 1):
            continue
        save_to_file(filled_contour)

    # Sort data by color
    data['trajectories'].sort(key = lambda x: COLORS_ORDER_IDX.index( x['color'] ))
    with open('./trjs.pickle', 'wb') as file:
        pickle.dump(data, file)

    # cv2.destroyAllWindows()
    # cv2.imshow('result', img_contours_global)
    cv2.imwrite('result.png', img_with_contour)
    cv2.waitKey(0)
