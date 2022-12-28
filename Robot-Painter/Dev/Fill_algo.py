import numpy as np
import cv2
import roboticstoolbox.tools.trajectory as rtb
import pickle

MIN_CONTOUR_AREA = 7

data = {
    'trajectories':[]
}

def img_to_square(img):
    x=y=max(img.shape[0], img.shape[1])
    if len(img.shape) == 3:
        square= np.ones((x,y,3))*(255,255,255)
    else: 
        square= np.zeros((x,y))
    square[int((y-img.shape[0])/2):int(y-(y-img.shape[0])/2), int((x-img.shape[1])/2):int(x-(x-img.shape[1])/2)] = img
    return square


img = img_to_square(cv2.imread('/home/leo/Downloads/Kozlova_art.jpeg'))
quantized_img = img_to_square(cv2.imread('./quantized_image.png'))
final_labels = img_to_square(np.loadtxt('segments.txt'))
COLORS = np.loadtxt('colors.txt', np.uint0)

compression_coeff = max(img.shape[0], img.shape[1])/400

result = np.zeros(img.shape, dtype=np.uint8)
cv2.imshow('quantized_img', quantized_img.astype(np.uint8))

def save_to_file(cnt, color):
    global data, compression_coeff
    cnt = np.array([point[0] for point in cnt])
    trajectory = cnt.copy()/compression_coeff
    if len(trajectory) == 0:
        return 
    trajectory[:, 0] = 400-trajectory[:, 0]-1
    trajectory /= 1000.0
    # trj = {'points': trajectory, 'width': 1.0, 'color': np.where(COLORS==color)[0][0]}

    trj_array = rtb.mstraj(trajectory, dt=0.02, qdmax=0.25, tacc=0.05)

    trj = {'points': trj_array.q, 'width': 1.0, 'color': np.where(COLORS==color)[0][0]}
    data['trajectories'].append(trj)

def find_nearest_point(control_point, cnt):
    min = np.inf
    index = -1
    for i, point in enumerate(cnt):
        error = np.linalg.norm(control_point - point, 2)
        if min > error:
            index = i
            min = error

    if min >= 10:
        return -1

    return index


def fill(mask, border, color_, brush=2):
    global result_path
    # result =np.zeros(img.shape)

    if cv2.contourArea(border) < MIN_CONTOUR_AREA:
        return

    # print(result_path.shape)
    if result_path.shape[0] != 0:
        starting_index = find_nearest_point(result_path[-1][0], border)
        if starting_index == -1:
            save_to_file(result_path, color_)
            result_path = border
        else:
            border = np.roll(border, -starting_index, axis=0)
            result_path = np.concatenate((result_path, border), axis=0)
    else:
        result_path = border

    for point in border:
        point = point[0]
        
        for r in range(-brush+1, brush):
            try: 
                # if point + np.array([0,r]) in int_mask:
                    result[point[1]+r][point[0]] = (color_[0], color_[1], color_[2])
                    mask[point[1] + r][point[0]] = 0
            except IndexError:
                pass
                # print("BORDER WARNING")
            try: 
                # if point + np.array([r,0]) in int_mask:
                    result[point[1]][point[0]+r] = (color_[0], color_[1], color_[2])
                    mask[point[1]][point[0] + r] = 0
            except IndexError:
                pass
                # print("BORDER WARNING")
            try: 
                # if point + np.array([r,r]) in int_mask:
                    result[point[1]+r][point[0]+r] = (color_[0], color_[1], color_[2])
                    mask[point[1] + r][point[0] + r] = 0
            except IndexError:
                pass
                # print("BORDER WARNING")
        cv2.imshow('result', result)
        cv2.imshow('amsk', mask)

        # cv2.waitKey(1)

            
   
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # print(f'[INFO]:\t Found {len(contours)} contours')
    # cv2.drawContours(mask_cnt, contours[0], -1, 255, thickness=-1)
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        # print(f"[INF^O]:\t Filling {i} contour")
        mask  = np.zeros(mask.shape, dtype=np.uint8)
        mask = cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
       
        fill(mask, cnt, color_, brush=brush)
        cv2.waitKey(1)
    
    save_to_file(result_path, color_)
    result_path = np.empty((0, 1, 2))


for l in np.unique(final_labels):
    if l == 0:
        continue
    mask = np.ones((quantized_img.shape[0], quantized_img.shape[1]), np.uint8) * 255
    mask[final_labels!=l] = 0
    border, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    border = border[0]
    region_colors = np.array([quantized_img[i][j] for i,j in zip(np.where(final_labels==l)[0], np.where(final_labels==l)[1])])
    clrs, indexes, counts = np.unique(region_colors, axis=0, return_index=True, return_counts=True)
    color_index = indexes[np.argmax(counts)]
    color_ = region_colors[color_index]
    result_path = np.empty((0, 1, 2))
    fill(mask, border, color_)

data['trajectories'].sort(key = lambda x: x['color'])
print(len(data['trajectories']))
with open('./trjs.pickle', 'wb') as file:
    pickle.dump(data, file)
cv2.imwrite('result_filled.png', result)
cv2.waitKey(0)
