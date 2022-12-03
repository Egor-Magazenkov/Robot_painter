import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from tkinter import filedialog, Tk



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
    zeros = cv2.resize(cv2.flip(cv2.cvtColor(cv2.imread('./result.png'), cv2.COLOR_BGR2RGB), 0), (400,400))
    canvas.imshow(zeros)
    cnt = 0
    for path in paths:
        path = convert_path(path)
        canvas.plot(path[:,0], path[:,1], linewidth=1, color='black')
        cnt += len(path)
    print(cnt)
    fig.canvas.draw()



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



canvas_size = 400
paths = []
filepath = ''
filename = ''
compression_coeff = None
img = None
img_width = None
canvas = None


def img_to_square(img):
    x=y=max(img.shape[0], img.shape[1])
    square= np.ones((x,y), np.uint8)*255
    square[int((y-img.shape[0])/2):int(y-(y-img.shape[0])/2), int((x-img.shape[1])/2):int(x-(x-img.shape[1])/2)] = img
    square = cv2.resize(square, (400, 400))
    return square


def start():
    global img_width, compression_coeff, img, gabor_res
    img = img_to_square(cv2.imread(filepath+filename, 0))
    # img = sharpening(img)
    img_width = img.shape[0]
    compression_coeff = img_width/canvas_size
    print(compression_coeff)
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

def update(val):
    global paths
    # fig.clf()
    canvas.cla()
    result=[]
    # result += gabor_res
    
    canny_val1 = canny_slider1.val
    canny_val2 = canny_slider2.val
    length = length_slider.val
    ksize = ksize_slider.val
    lambd = lambda_slider.val
    blur = blur_slider.val
    sigma = sigma_slider.val

    g_kernel_1 = cv2.getGaborKernel((ksize, ksize), sigma, 0, lambd, 0.5, np.pi)
    g_kernel_2 = cv2.getGaborKernel((ksize, ksize), sigma, np.pi/2, lambd, 0.5, np.pi)
    g_kernel_3 = cv2.getGaborKernel((ksize, ksize), sigma, np.pi/4, lambd, 0.5, np.pi)
    g_kernel_4 = cv2.getGaborKernel((ksize, ksize), sigma, 3*np.pi/4, lambd, 0.5, np.pi)

    img1 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_1)
    img2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_2)
    img3 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_3)
    img4 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel_4)

    img12 = cv2.bitwise_xor(img2,img1)
    img34 = cv2.bitwise_xor(img3,img4)
    img1234 = cv2.bitwise_xor(img12, img34)
    thinned = cv2.ximgproc.thinning(img1234)
    res,hier = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    gabor_res = []
    for cnt in res:
        if cv2.arcLength(cnt, True) < length:
            continue
        c = cv2.approxPolyDP(cnt, 10**(-10)*cv2.arcLength(cnt, True), True)
        # c = np.array([c[i] for i in range(len(c)) if np.linalg.norm(c[i-1]-c[i]) > 2])
        if len(c) >0:
            gabor_res.append(c)
    gabor_res = [np.array([point[0] for point in cnt]) for cnt in gabor_res]
    result += gabor_res
    image = image_processing(img.copy(), blur=blur)

    
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

        c = cv2.approxPolyDP(cnt, 10**(-10)*cv2.arcLength(cnt, True), True)
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
ksize_slider.on_changed(update)
lambda_slider.on_changed(update)
blur_slider.on_changed(update)
sigma_slider.on_changed(update)

open_button_axes = plt.axes([0.2, 0.2, 0.2, 0.1])
open_button = Button(open_button_axes, 'Открыть',color="cyan")
def open_file(val):
    global filename, filepath
    Tk().withdraw()
    path_to_file = filedialog.askopenfilename(title = "Выберите фото",\
        filetypes=(("Фотографии", "*.jpg"), ("Фотографии", "*.png"), ("Фотографии", "*.jpeg")))
    filename = path_to_file.split('/')[-1]
    filepath = path_to_file.split(filename)[0]
    start()
open_button.on_clicked(open_file)


submit_button_axes = plt.axes([0.6, 0.2, 0.2, 0.1])
submit_button = Button(submit_button_axes, 'Сохранить',color="green")
def submit(val): 
    fig.savefig(filepath + filename.split('.')[0] + 'res.png')       
    with open(filepath + filename.split('.')[0] + '.txt', 'w') as file:
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
        
    result = {'paths':[]}
    def to_json(path):
        trj = {'points': []}
        path=convert_path(path)/1000
        for point in path:
            trj['points'].append({'p':list(point), 'width':1.0})
        return trj
    for path in paths:
        # path = convert_path(path)     
        result['paths'].append(to_json(path))
    with open(filepath + filename.split('.')[0] + '.json', 'w') as file:
        file.write(json.dumps(result, indent=4))
submit_button.on_clicked(submit)

plt.show()