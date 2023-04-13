"""Code for extracting features from image in bezier curves"""
from tkinter import filedialog, Tk
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
import roboticstoolbox.tools.trajectory as rtb


def get_bezier_coef(points):
    """Connect ordered set of points with C2 continuous bezier curves."""
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n_points = len(points) - 1

    # build coefficents matrix
    coeffs = 4 * np.identity(n_points)
    np.fill_diagonal(coeffs[1:], 1)
    np.fill_diagonal(coeffs[:, 1:], 1)
    coeffs[0, 0] = 2
    coeffs[n_points - 1, n_points - 1] = 7
    coeffs[n_points - 1, n_points - 2] = 2

    # build points vector
    res_points = [2 * (2 * points[i] + points[i + 1]) for i in range(n_points)]
    res_points[0] = points[0] + 2 * points[1]
    res_points[n_points - 1] = 8 * points[n_points - 1] + points[n_points]

    # solve system, find a & b
    control_point_1 = np.linalg.solve(coeffs, res_points)
    control_point_2 = [0] * n_points
    for i in range(n_points - 1):
        control_point_2[i] = 2 * points[i + 1] - control_point_1[i + 1]
    control_point_2[n_points - 1] = (control_point_1[n_points - 1] + points[n_points]) / 2

    return points[:-1], control_point_1, control_point_2, points[1:]

def evaluate_bezier(points, n_points, i):
    """Generate sets of n_points points from cubic bezier coeffitions."""
    # returns the general Bezier cubic formula given 4 control points
    def get_cubic(cp_1, cp_2, cp_3, cp_4):
        return lambda t: \
                np.power(1 - t, 3) * cp_1 + \
                3 * np.power(1 - t, 2) * t * cp_2 + \
                3 * (1 - t) * np.power(t, 2) * cp_3 + \
                np.power(t, 3) * cp_4

    # return one cubic curve for each consecutive points
    def get_bezier_cubic(points, i):
        _, control_point_1, control_point_2, _ = get_bezier_coef(points)
        return [
            get_cubic(points[i], control_point_1[i], control_point_2[i], points[i + 1])
            for i in range(len(points) - 1)
        ]

    curves = get_bezier_cubic(points, i)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n_points)])

def convert_path(path):
    """Convert path to fit picture on canvas."""
    res = path/COMPRESSION_COEFF
    # flip image
    res[:, 1] = CANVAS_SIZE - res[:, 1] % CANVAS_SIZE
    res = np.round(res/2, 1) * 2
    res = np.array([res[i] for i in sorted(np.unique(res, return_index=True, axis=0)[1])])
    return res

def draw_paths(points):
    """Plot points onto the canvas."""
    canvas.add_patch(Rectangle((0, 0), CANVAS_SIZE, CANVAS_SIZE, fill = False))
    cnt = 0
    for path in points:
        path = convert_path(path)
        canvas.plot(path[:,0], path[:,1], linewidth=1, color='black')
        cnt += len(path)
    fig.canvas.draw()

def image_processing(src,  smooth=95, scale=192, blur=5, sigma=5):
    """Preprocess image to get rid of glitches, shadows and noises."""
    def image_transform(src, smooth=smooth, scale=scale):
        smooth = cv2.GaussianBlur(src, (smooth,smooth), 0)
        division = cv2.divide(src, smooth, scale=scale)
        # cv2.imshow('s', division)
        # cv2.waitKey(0)
        return division

    def adjust_brightness(src):
        cols, rows = src.shape
        brightness = np.sum(src) / (255 * cols * rows)
        min_br = 0.8
        ratio = brightness / min_br
        if ratio >= 1:
            print("Image already bright enough")
            return src
        return cv2.convertScaleAbs(src, alpha = 1 / ratio, beta = 0)

    res = image_transform(src)
    res = adjust_brightness(res)
    res = cv2.GaussianBlur(res, (blur,blur), sigma)
    return res



CANVAS_SIZE = 400
paths = []
FILEPATH = ''
FILENAME = ''
COMPRESSION_COEFF = None
IMAGE = None
IMG_WIDTH = None


def img_to_square(src):
    """Refactor image to square size with bigger side as side of square."""
    height = width = max(src.shape[0], src.shape[1])
    square= np.ones((height,width), np.uint8)*255
    square[
            int((width-src.shape[0])/2):int(width-(width-src.shape[0])/2),
            int((height-src.shape[1])/2):int(height-(height-src.shape[1])/2)
            ] = src
    square = cv2.resize(square, (CANVAS_SIZE, CANVAS_SIZE))
    return square


def start():
    """Begin the image processing algos."""
    global IMG_WIDTH, COMPRESSION_COEFF, IMAGE
    IMAGE = img_to_square(cv2.imread(FILEPATH+FILENAME, 0))
    # IMAGE = sharpening(IMAGE)
    IMG_WIDTH = IMAGE.shape[0]
    COMPRESSION_COEFF = IMG_WIDTH/CANVAS_SIZE
    print(f'Compressing the image {COMPRESSION_COEFF} times')
    update(100)

fig, canvas = plt.subplots(figsize = (10, 10))
# canvas.axis('off')

fig_1, ax_1 = plt.subplots(figsize=(5, 10))
ax_1.axis('off')

canny_axes1 = plt.axes([0.25, 0.9, 0.5, 0.02])
canny_slider1 = Slider(canny_axes1, "Canny 1", 10, 250, 130, valstep = 5)
canny_axes2 = plt.axes([0.25, 0.8, 0.5, 0.02])
canny_slider2 = Slider(canny_axes2, "Canny", 10, 250, 250, valstep = 5)
length_axes = plt.axes([0.25, 0.7, 0.5, 0.02])
length_slider = Slider(length_axes, "length", 0, 250, 50, valstep = 5)
ksize_axes = plt.axes([0.25, 0.6, 0.5, 0.02])
ksize_slider = Slider(ksize_axes, "Ksize", 1, 200, 30, valstep = 2)
lambda_axes = plt.axes([0.25, 0.5, 0.5, 0.02])
lambda_slider = Slider(lambda_axes, "lambda", 1, 200, 7, valstep = 2)
blur_axes = plt.axes([0.25, 0.4, 0.5, 0.02])
blur_slider = Slider(blur_axes, "Blur", 1, 9, 5, valstep = 2)
sigma_axes = plt.axes([0.25, 0.3, 0.5, 0.02])
sigma_slider = Slider(sigma_axes, "Sigma", 1, 200, 5, valstep = 2)

def gabor_filter(ksize, sigma, lambda_, length):
    """Extract features from image using Gabor filter."""
    img1 = cv2.filter2D(IMAGE, cv2.CV_8UC3,
                        cv2.getGaborKernel((ksize, ksize), sigma, 0, lambda_, 0.5, np.pi))
    img2 = cv2.filter2D(IMAGE, cv2.CV_8UC3,
                        cv2.getGaborKernel((ksize, ksize), sigma, np.pi/2, lambda_, 0.5, np.pi))
    img3 = cv2.filter2D(IMAGE, cv2.CV_8UC3,
                        cv2.getGaborKernel((ksize, ksize), sigma, np.pi/4, lambda_, 0.5, np.pi))
    img4 = cv2.filter2D(IMAGE, cv2.CV_8UC3,
                        cv2.getGaborKernel((ksize, ksize), sigma, 3*np.pi/4, lambda_, 0.5, np.pi))

    img1234 = cv2.bitwise_xor(cv2.bitwise_xor(img2,img1), cv2.bitwise_xor(img3,img4))
    thinned = cv2.ximgproc.thinning(img1234)
    res, _ = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    gabor_res = []
    for cnt in res:
        if cv2.arcLength(cnt, True) < length:
            continue
        contour = cv2.approxPolyDP(cnt, 10**(-10)*cv2.arcLength(cnt, True), True)
        # c = np.array([c[i] for i in range(len(c)) if np.linalg.norm(c[i-1]-c[i]) > 2])
        if len(contour) > 0:
            gabor_res.append(contour)
    gabor_res = [np.array([point[0] for point in cnt]) for cnt in gabor_res]
    return gabor_res

def update(val):
    """Read the values from sliders, apply all algos and draw result onto canvas."""
    global paths
    # fig.clf()
    canvas.cla()
    result=[]

    length = length_slider.val

    result += gabor_filter(ksize_slider.val, sigma_slider.val, lambda_slider.val, length)

    image = image_processing(IMAGE.copy(), blur=blur_slider.val)

    canny = cv2.Canny(image, canny_slider1.val, canny_slider2.val,
                                apertureSize=3, L2gradient = True)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)))

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    hierarchy = list(hierarchy[0])

    cnts_to_del = []
    for i, cnt in enumerate(contours):
        if cv2.arcLength(cnt, True) < length:
            continue

        if i in cnts_to_del:
            continue

        if hierarchy[i][2] != -1:
            cnts_to_del.append(hierarchy[i][2])
            k = hierarchy[hierarchy[i][2]][0]
            while k != -1:
                cnts_to_del.append(k)
                k = hierarchy[k][0]

        contour = cv2.approxPolyDP(cnt, 10**(-10)*cv2.arcLength(cnt, True), True)
        # c = np.array([c[i] for i in range(len(c)) if np.linalg.norm(c[i-1]-c[i]) > 2])
        contour = [point[0] for point in contour]
        if not contour:
            continue
        # c.append(c[0])
        contour = np.array(contour)
        result.append(contour)
    draw_paths(result)
    paths=result

canny_slider1.on_changed(update)
canny_slider2.on_changed(update)
length_slider.on_changed(update)
ksize_slider.on_changed(update)
lambda_slider.on_changed(update)
blur_slider.on_changed(update)
sigma_slider.on_changed(update)

open_button_axes = plt.axes([0.2, 0.2, 0.2, 0.1])
open_button = Button(open_button_axes, 'Открыть',color="cyan")

def open_file(val):
    """Handle the name of the file from dialog input."""
    global FILENAME, FILEPATH
    Tk().withdraw()
    path_to_file = filedialog.askopenfilename(title = "Выберите фото",\
        filetypes=(("Фотографии", "*.jpg"), ("Фотографии", "*.png"), ("Фотографии", "*.jpeg")))
    FILENAME = path_to_file.split('/')[-1]
    FILEPATH = path_to_file.split(FILENAME)[0]
    start()
open_button.on_clicked(open_file)


submit_button_axes = plt.axes([0.6, 0.2, 0.2, 0.1])
submit_button = Button(submit_button_axes, 'Сохранить',color="green")

def submit(val):
    """Save the result canvas, trajectories and  sliders' values on triiger of SUBMIT button."""
    fig.savefig(FILEPATH + FILENAME.split('.')[0] + 'res.png')
    with open(FILEPATH + FILENAME.split('.')[0] + '.txt', 'w',encoding="utf8") as file:
        canny_val1 = canny_slider1.val
        canny_val2 = canny_slider2.val
        length = length_slider.val
        ksize = ksize_slider.val
        lambd = lambda_slider.val
        blur = blur_slider.val
        sigma = sigma_slider.val
        file.write("canny_1\t" + str(canny_val1) + "\n")
        file.write("canny_2\t" + str(canny_val2) + "\n")
        file.write("length\t"+ str(length)+ "\n")
        file.write("blur\t"+ str(blur)+ "\n")
        file.write("ksize\t"+ str(ksize)+ "\n")
        file.write("sigma\t"+ str(sigma)+ "\n")
        file.write("lambda\t"+ str(lambd)+ "\n")
    with open('./trjs.pickle', 'rb') as file:
        result = pickle.load(file)
        for i, trj in reversed(list(enumerate(result['trajectories']))):
            if trj['color'] == 0:
                result['trajectories'].pop(i)
            if trj['color'] != 0:
                break
    def to_pickle(path):
        path=convert_path(path)
        path[:, 1] = CANVAS_SIZE - path[:, 1] % CANVAS_SIZE
        path_array = rtb.mstraj(path/1000, dt=0.002, qdmax=0.25, tacc=0.05)
        trj = {'points': path_array.q, 'width': 1.0, 'color': 0}
        return trj
    for path in paths:
        result['trajectories'].append(to_pickle(path))

    with open('./trjs.pickle', 'wb') as file:
        pickle.dump(result, file)
submit_button.on_clicked(submit)

plt.show()
