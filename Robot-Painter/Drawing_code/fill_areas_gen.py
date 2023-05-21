"""Algorithm for spiral filling areas."""
import sys
import pickle
import numpy as np
import cv2
import roboticstoolbox.tools.trajectory as rtb

from paint import Painter

MIN_CONTOUR_AREA = 7

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

compression_coeff = max(img.shape[0], img.shape[1])/400
result = np.zeros(img.shape, dtype=np.uint8)
#cv2.imshow('quantized_img', quantized_img.astype(np.uint8))

def save_to_file(cnt, color, c_coeff=compression_coeff):
    """Convert path to fit picture on canvas and put it into result dictionary."""
    global data
    cnt = np.array([point[0] for point in cnt])
    trajectory = cnt.copy() / c_coeff
    if len(trajectory) == 0:
        return
    #trajectory[:, 0] = 400 - trajectory[:, 0] - 1
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
    global result_path, data
    # result =np.zeros(img.shape)

    if cv2.contourArea(boundary) < MIN_CONTOUR_AREA:
        return

    # TODO 
    painter = Painter({'image': cv2.bitwise_and(quantized_img,quantized_img,mask = mask_image), 'threshold': 0.05, 'brush_sizes': [5], 'maxLength': 45, 'minLength': 5, 'grid_fac': 1, 'filter_fac': 1, 'length_fac':1, 'blur_fac': 0.5})
    
    data['trajectories'].extend(painter.getData())
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

#data['trajectories'].sort(key = lambda x: x['color'])
print(len(data['trajectories']))
with open('./trjs.pickle', 'wb') as file:
    pickle.dump(data, file)
cv2.imwrite('result_filled.png', result)
cv2.waitKey(0)
