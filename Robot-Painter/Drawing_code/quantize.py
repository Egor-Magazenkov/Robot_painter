from skimage import segmentation, color
import numpy as np
import cv2
import scipy.spatial as sp


def plot_palette(colors, figname = 'colors pallete'):
    pallete = np.tile(colors[0], (100, 100, 1))

    for i in range(1, len(colors)):
        pallete = np.concatenate((pallete, np.tile(colors[i], (100, 100, 1))), axis=1)

    np.savetxt('colors.txt', colors, fmt='%d')
    cv2.imshow(figname, pallete)
    cv2.imwrite('colors_pallete.png', pallete)


def find_nearest_color(colors):
    global main_colors

    tree = sp.KDTree(main_colors)
    idx = []
    for c in colors:
        data = tuple(c)
        _, result = tree.query(data)
        idx.append(result)

    return idx


def generate_colors_order(colors):
    colors_hsv = -cv2.cvtColor(colors, cv2.COLOR_BGR2HSV)[0]
    return colors_hsv[:, 2].argsort()


if __name__ == '__main__':
    img = cv2.imread('/home/leo/Downloads/stable_diffusion_ex.png')
    img = cv2.bilateralFilter(img, 7, 21,21)
    cv2.imshow('img', img)

    # x=y=max(img.shape[0], img.shape[1])
    # square= np.ones((x,y,3), np.uint8)*255
    # square[int((y-img.shape[0])/2):int(y-(y-img.shape[0])/2), int((x-img.shape[1])/2):int(x-(x-img.shape[1])/2)] = img
    # img_squared = cv2.resize(square, (400,400))

    segments = segmentation.slic(img, sigma=3, compactness=20, n_segments=100, start_label=1, convert2lab=True, max_num_iter=100)
    blank_img = np.ones(img.shape)*255

    out1 = color.label2rgb(segments,  img, kind = 'avg', bg_label = 0)
    out1 = out1.astype(np.uint8)

    img_with_branch = segmentation.mark_boundaries(blank_img, segments)
    img_with_branch_2 = segmentation.mark_boundaries(out1, segments)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    Z = img_LAB.reshape((-1,3))
    Z = np.float32(Z)

    ret,label,center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    label = label.reshape((img.shape[0], img.shape[1]))
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res3 = cv2.cvtColor(res2, cv2.COLOR_Lab2BGR)

    center = cv2.cvtColor(np.array([center], np.uint8), cv2.COLOR_LAB2BGR)
    idx = generate_colors_order(center)
    plot_palette(center[0][idx], 'centers')

    
    cv2.imshow('Quantized image', res3)
    cv2.imshow('Contours on image', img_with_branch_2)
    cv2.imwrite('./quantized_image.png', res3)
    cv2.imwrite('./mask.jpg', out1)
    cv2.imwrite('./branch.jpg', img_with_branch)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
