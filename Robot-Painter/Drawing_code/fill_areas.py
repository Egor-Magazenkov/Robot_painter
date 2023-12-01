"""Algorithm for spiral filling areas."""
import sys
import pickle
import numpy as np
import cv2
import roboticstoolbox.tools.trajectory as rtb

MIN_CONTOUR_AREA = 7
CANVAS_SIZE = 300

data = {
    'trajectories':[]
}
result_path = np.empty((0, 1, 2))

def img_to_square(src):
    """Refactor image to square size with bigger side as side of square."""
    height = width = max(src.shape[0], src.shape[1])
    if len(src.shape) == 3:
        square= np.ones((height, width, 3))*(255,255,255)
    else:
        square= np.zeros((height, width))
    square[int((width-src.shape[0])/2):int(width-(width-src.shape[0])/2),
            int((height-src.shape[1])/2):int(height-(height-src.shape[1])/2)] = src
    return square

if len(sys.argv) != 5:
    print('Bad argument usage.\n'
                 'Usage: python3 fill_areas.py ' 
                                '<path/to/image> '
                                '<path/to/quantized_image> '
                                '<path/to/segments.txt> '
                                '<path/to/colors.txt> ')
    sys.exit(1)
else:
    img = img_to_square(cv2.imread(sys.argv[1]))
    quantized_img = img_to_square(cv2.imread(sys.argv[2]))
    final_labels = img_to_square(np.loadtxt(sys.argv[3]))
    COLORS = np.loadtxt(sys.argv[4], np.uint0)
compression_coeff = max(img.shape[0], img.shape[1])/CANVAS_SIZE

result = np.zeros(img.shape, dtype=np.uint8)
cv2.imshow('quantized_img', quantized_img.astype(np.uint8))

def save_to_file(cnt, color, c_coeff=compression_coeff):
    """Convert path to fit picture on canvas and put it into result dictionary."""
    global data
    cnt = np.array([point[0] for point in cnt])
    trajectory = cnt.copy() / c_coeff
    if len(trajectory) == 0:
        return
    #trajectory[:, 0] = 400 - trajectory[:, 0] - 1
    trajectory[:, 1] = CANVAS_SIZE - trajectory[:, 1] % CANVAS_SIZE
    trajectory /= 1000.0
    # trj = {'points': trajectory, 'width': 1.0, 'color': np.where(COLORS==color)[0][0]}

    trj_array = rtb.mstraj(trajectory, dt=0.02, qdmax=0.25, tacc=0.05)

    trj = {'points': trj_array.q, 'width': 1.0, 'color': np.where(COLORS==color)[0][0]}
    data['trajectories'].append(trj)

def find_nearest_point(control_point, cnt):
    """Get index of the nearest point of contour to control_point."""
    min_ = np.inf
    index = -1
    for i, point in enumerate(cnt):
        error = np.linalg.norm(control_point - point, 2)
        if min_ > error:
            index = i
            min_ = error

    if min_ >= 10:
        return -1

    return index

def fill(mask_image, boundary, color, brush=3):
    """Generate spiral or spirals to fill area bounded by border."""
    global result_path
    # result =np.zeros(img.shape)

    if cv2.contourArea(boundary) < MIN_CONTOUR_AREA:
        return

    # print(result_path.shape)
    if result_path.shape[0] != 0:
        starting_index = find_nearest_point(result_path[-1][0], boundary)
        if starting_index == -1:
            save_to_file(result_path, color)
            result_path = boundary
        else:
            boundary = np.roll(boundary, -starting_index, axis=0)
            result_path = np.concatenate((result_path, boundary), axis=0)
    else:
        result_path = boundary

    for point in boundary:
        point = point[0]
        for radius in range(-brush+1, brush):
            try:
                # if point + np.array([0,r]) in int_mask:
                result[point[1]+radius][point[0]] = (color[0], color[1], color[2])
                mask_image[point[1] + radius][point[0]] = 0
            except IndexError:
                pass
                # print("BORDER WARNING")
            try:
                # if point + np.array([r,0]) in int_mask:
                result[point[1]][point[0]+radius] = (color[0], color[1], color[2])
                mask_image[point[1]][point[0] + radius] = 0
            except IndexError:
                pass
                # print("BORDER WARNING")
            try:
                # if point + np.array([r,r]) in int_mask:
                result[point[1]+radius][point[0]+radius] = (color[0], color[1], color[2])
                mask_image[point[1] + radius][point[0] + radius] = 0
            except IndexError:
                pass
                # print("BORDER WARNING")
        cv2.imshow('result', result)
        # cv2.imshow('amsk', mask_image)

        # cv2.waitKey(1)

    contours, _ = cv2.findContours(mask_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # print(f'[INFO]:\t Found {len(contours)} contours')
    # cv2.drawContours(mask_cnt, contours[0], -1, 255, thickness=-1)

    for _, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        # print(f"[INF^O]:\t Filling {i} contour")
        mask_image  = np.zeros(mask_image.shape, dtype=np.uint8)
        mask_image = cv2.drawContours(mask_image, [cnt], -1, 255, thickness=cv2.FILLED)
        fill(mask_image, cnt, color, brush=brush)
        cv2.waitKey(1)
    save_to_file(result_path, color)
    result_path = np.empty((0, 1, 2))


for l in np.unique(final_labels.flatten()):
    if l == 0:
        continue
    mask = np.ones((quantized_img.shape[0], quantized_img.shape[1]), np.uint8) * 255
    mask[final_labels!=l] = 0
    border, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    border = border[0]
    region_colors = np.array([quantized_img[i][j] for i,j in
        zip(np.where(final_labels==l)[0], np.where(final_labels==l)[1])])
    clrs, indexes, counts = np.unique(region_colors, axis=0, return_index=True, return_counts=True)
    color_index = indexes[np.argmax(counts)]
    color_ = region_colors[color_index]
    # color_ = quantized_img[border[0][0][1]][border[0][0][0]]
    result_path = np.empty((0, 1, 2))
    fill(mask, border, color_)

data['trajectories'].sort(key = lambda x: x['color'])
print(len(data['trajectories']))
with open('./trjs.pickle', 'wb') as file:
    pickle.dump(data, file)
cv2.imwrite('result_filled.png', result)
cv2.waitKey(0)
