import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
import mediapipe as mp
import math
from tkinter import filedialog, Tk


def image_processing(img,  smooth=95, scale=192, blur=5, sigma=5):
    def image_transform(img, smooth=95, scale=192):
        smooth = cv2.GaussianBlur(img, (smooth,smooth), 0)
        division = cv2.divide(img, smooth, scale=250)
        # cv2.imshow('s', division)
        # cv2.waitKey(0)
        return division
    
    def adjust_brightness(img):
        cols, rows = img.shape
        brightness = np.sum(img) / (255 * cols * rows)
        min_br = 0.8
        ratio = brightness / min_br
        if ratio >= 1:
            print("Image already bright enough")
            return img
        return cv2.convertScaleAbs(img, alpha = 1 / ratio, beta = 0)

    img = image_transform(img)
    img = adjust_brightness(img)
    img = cv2.GaussianBlur(img, (blur,blur), sigma)
    return img

def sharpening(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def find_face_parts(img):
    mp_face_mesh = mp.solutions.face_mesh

    face_parts_ = []
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                idx_to_coordinates = {}
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmark_px = (min(math.floor(landmark.x * image.shape[0]), image.shape[0] - 1), min(math.floor(landmark.y * image.shape[1]), image.shape[1] - 1))
                    if landmark_px:
                        idx_to_coordinates[idx] = landmark_px
                
                def convert_points(f):
                    f = evaluate_bezier(f, 4, -1)
                    f[:,0] = f[:,0]-2*f[:,0]+img.shape[0]
                    return f

                face_oval = convert_points(np.array([idx_to_coordinates[i] for i in [389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127]]))
                left_eye = convert_points(np.array([idx_to_coordinates[i] for i in [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]]))
                right_eye = convert_points(np.array([idx_to_coordinates[i] for i in  [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249,263]]))
                inner_lip = convert_points(np.array([idx_to_coordinates[i] for i in  [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78]]))
                outer_lip = convert_points(np.array([idx_to_coordinates[i] for i in  [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61]]))
                left_eyebrow = convert_points(np.array([idx_to_coordinates[i] for i in  [55,65,52,53,46]]))
                right_eyebrow = convert_points(np.array([idx_to_coordinates[i] for i in  [285,295,282,283,276]]))
                nose = convert_points(np.array([idx_to_coordinates[i] for i in  [122,174,198,49,64, 59, 60,20,242,94,462,250,290,289,294,279,420,399,351]]))
                additional_nose = convert_points(np.array([idx_to_coordinates[i] for i in  [19,1,4]]))
                face_parts_ += [additional_nose,nose,face_oval, left_eye, right_eye, inner_lip, outer_lip, left_eyebrow, right_eyebrow]
        else:
            print("NO FACE")
    return face_parts_

# find the a & b points
def get_bezier_coef(points):
        # since the formulas work given that we have n+1 points
        # then n must be this:
        n = len(points) - 1

        # build coefficents matrix
        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2

        # build points vector
        P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        P[0] = points[0] + 2 * points[1]
        P[n - 1] = 8 * points[n - 1] + points[n]

        # solve system, find a & b
        A = np.linalg.solve(C, P)
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2

        return points[:-1], A, B, points[1:]
# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n, i):
    # returns the general Bezier cubic formula given 4 control points
    def get_cubic(a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

    # return one cubic curve for each consecutive points
    def get_bezier_cubic(points, i):
        _, A, B,_ = get_bezier_coef(points)
        return [
            get_cubic(points[i], A[i], B[i], points[i + 1])
            for i in range(len(points) - 1)
        ]

    curves = get_bezier_cubic(points, i)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])

def convert_path(path):
    res = path/compression_coeff
    # flip image
    res[:,1] = canvas_size - res[:,1]%canvas_size
    res = np.round(res/2, 1)*2
    res = np.array([res[i] for i in sorted(np.unique(res, return_index=True, axis=0)[1])])
    return res
    
def draw_paths(paths):
    canvas.add_patch(Rectangle((0, 0), canvas_size, canvas_size, fill = False))
    cnt = 0
    for path in paths:
        path = convert_path(path)
        canvas.plot(path[:,0], path[:,1], linewidth=1, color='black')
        cnt += len(path)
    print(cnt)
    fig.canvas.draw()


canvas_size = 330
paths = []
result_path = 'Code/Result_Dead/'
filepath = ''
filename = ''
compression_coeff = None
img = None
img_width = None
canvas = None
face_elements = None

def img_to_square(img):
    x=y=max(img.shape[0], img.shape[1])
    square= np.ones((x,y), np.uint8)*255
    square[int((y-img.shape[0])/2):int(y-(y-img.shape[0])/2), int((x-img.shape[1])/2):int(x-(x-img.shape[1])/2)] = img
    return square

def start():
    global img_width, compression_coeff, face_elements, img, gabor_res
    img = img_to_square(cv2.imread(filepath+filename, 0))
    # img = cv2.resize(cv2.imread(filepath+filename, 0), (canvas_size, canvas_size))
    # img = sharpening(img)
    face_elements = find_face_parts(img)
    img_width = img.shape[0]
    compression_coeff = img_width/canvas_size
    print(compression_coeff)
    
    
    g_kernel_1 = cv2.getGaborKernel((33, 33), 7, 0, 22, 0.5, np.pi)
    g_kernel_2 = cv2.getGaborKernel((33, 33), 7, np.pi/2, 22, 0.5, np.pi)
    g_kernel_3 = cv2.getGaborKernel((33, 33), 7, np.pi/4, 22, 0.5, np.pi)
    g_kernel_4 = cv2.getGaborKernel((33, 33), 7, 3*np.pi/4, 22, 0.5, np.pi)

    img1 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_1)
    img2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_2)
    # img3 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_3)
    # img4 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_4)

    img12 = cv2.bitwise_xor(img2,img1)
    # img34 = cv2.bitwise_xor(img3,img4)
    # img1234 = cv2.bitwise_xor(img12, img34)
    thinned = cv2.ximgproc.thinning(img12)
    res,hier = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    gabor_res = []
    for cnt in res:
        if cv2.arcLength(cnt, True) < 50:
            continue
        c = cv2.approxPolyDP(cnt, 10**(-4)*cv2.arcLength(cnt, True), True)
        # c = np.array([c[i] for i in range(len(c)) if np.linalg.norm(c[i-1]-c[i]) > 2])
        if len(c) >0:
            gabor_res.append(c)
    gabor_res = [np.array([point[0] for point in cnt]) for cnt in gabor_res]
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
length_slider = Slider(length_axes, "length", 0, 250, 50, valstep = 10)
smooth_axes = plt.axes([0.25, 0.6, 0.5, 0.02])
smooth_slider = Slider(smooth_axes, "Smooth", 30, 200, 95, valstep = 10)
scale_axes = plt.axes([0.25, 0.5, 0.5, 0.02])
scale_slider = Slider(scale_axes, "Scale", 100, 350, 200, valstep = 10)
blur_axes = plt.axes([0.25, 0.4, 0.5, 0.02])
blur_slider = Slider(blur_axes, "Blur", 1, 9, 5, valstep = 2)
sigma_axes = plt.axes([0.25, 0.3, 0.5, 0.02])
sigma_slider = Slider(sigma_axes, "Sigma", 1, 9, 5, valstep = 2)

def update(val):
    global paths
    # fig.clf()
    canvas.cla()
    result=[]
    result += face_elements
    result += gabor_res
    
    canny_val1 = canny_slider1.val
    canny_val2 = canny_slider2.val
    length = length_slider.val
    smooth = smooth_slider.val
    scale = scale_slider.val
    blur = blur_slider.val
    sigma = sigma_slider.val
    image = image_processing(img.copy(), smooth, scale, blur, sigma)

    canny = cv2.Canny(image, canny_val1, canny_val2, apertureSize=3, L2gradient = True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

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

        c = cv2.approxPolyDP(cnt, 10**(-4)*cv2.arcLength(cnt, True), True)
        # c = np.array([c[i] for i in range(len(c)) if np.linalg.norm(c[i-1]-c[i]) > 2])
        c = [point[0] for point in c]
        if not c:
            continue
        # c.append(c[0])
        c = np.array(c)
        result.append(c)
    draw_paths(result)
    paths=result

canny_slider1.on_changed(update)
canny_slider2.on_changed(update)
length_slider.on_changed(update)
smooth_slider.on_changed(update)
scale_slider.on_changed(update)
blur_slider.on_changed(update)
sigma_slider.on_changed(update)

submit_button_axes = plt.axes([0.6, 0.2, 0.2, 0.1])
submit_button = Button(submit_button_axes, 'Сохранить',color="green")
def submit(val):
    result = {"canvas":{"width":canvas_size, "height":canvas_size, "dpi":96}, \
        "colors":[{"r":240,"g":80,"b":100}, {"r":20,"g":60,"b":240}, {"r":30,"g":50,"b":140}]}
    result["strokes"] = []
    def to_json(path):
        p_0, p_1, p_2, p_3 = get_bezier_coef(path)
        
        res = []
        for i in range(len(p_0)):
            stroke = {}
            stroke["type"] = "cubic_bezier"
            stroke["p0"] = {"x":p_0[i][0], "y":p_0[i][1]}
            stroke["p1"] = {"x":p_1[i][0], "y":p_1[i][1]}
            stroke["p2"] = {"x":p_2[i][0], "y":p_2[i][1]}
            stroke["p3"] = {"x":p_3[i][0], "y":p_3[i][1]}
            stroke["width"] = 20.4
            stroke["color_id"] = 0
            res.append(stroke)
        return res
        
    for path in paths:
        path=convert_path(path)
        if len(path) > 3:
            result['strokes']+= to_json(path)
    with open(result_path + filename.split('.')[0] + '.json', 'w') as file:
        file.write(json.dumps(result, indent=4))
submit_button.on_clicked(submit)

open_button_axes = plt.axes([0.2, 0.2, 0.2, 0.1])
open_button = Button(open_button_axes, 'Открыть',color="cyan")
def open_file(val):
    global filename, filepath
    Tk().withdraw()
    path_to_file = filedialog.askopenfilename(initialdir = "/home/leo/Downloads/",title = "Выберите фото",\
        filetypes=(("Фотографии", "*.jpg"), ("Фотографии", "*.png"), ("Фотографии", "*.jpeg")))
    filename = path_to_file.split('/')[-1]
    filepath = path_to_file.split(filename)[0]
    start()
open_button.on_clicked(open_file)

plt.show()


# img = cv2.imread('/home/leo/Downloads/boitsev.png', 0)

# cv2.imshow('img',img)
# cv2.imshow('', image_processing(img.copy()))
# cv2.waitKey(0)