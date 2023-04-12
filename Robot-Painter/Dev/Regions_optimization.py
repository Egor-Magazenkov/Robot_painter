import cv2
import numpy as np
from skimage import segmentation, color

def find_nearest_color(color):
    global colors
    diff = np.linalg.norm(np.array(color) - colors, axis=1)
    return np.argmin(diff)

def region_color(labels, num):
    global quantized_img
    region_colors = np.array([quantized_img[i][j] for i,j in zip(np.where(labels==num)[0], np.where(labels==num)[1])])
    # print(np.unique(region_colors, axis=0,return_index=True, return_counts=True))
    clrs, indexes, counts = np.unique(region_colors, axis=0, return_index=True, return_counts=True)
    color_index = indexes[np.argmax(counts)]
    return region_colors[color_index]

img = cv2.imread('/home/leo/Downloads/pic2.jpeg')
quantized_img = cv2.imread('./quantized_image.png')
colors = np.loadtxt('colors.txt')
# labels = segmentation.slic(img, sigma=3, compactness=20, n_segments=1100, start_label=1, convert2lab=True, max_num_iter=100)
labels = np.loadtxt('segments.txt', dtype=np.uint0)
print(labels)
print(len(np.unique(labels.flatten())))
superpixels = color.label2rgb(labels, quantized_img, kind='avg')

labels_optimized = labels.copy()
res = img.copy()
print(img.shape)
print(labels.shape)
for j in range(img.shape[0]):
    for i in range(img.shape[1]):
        current_label = labels[j][i]
        current_color = superpixels[j][i]
        res[j][i] = current_color
        try:
            if labels[j][i+1] > current_label and np.array_equal(superpixels[j][i+1], current_color):
                labels[j][i+1] = current_label
                superpixels[j][i+1] = current_color
        except IndexError:
                pass # print("BORDER WARNING")
        try:
            if labels[j+1][i] > current_label and np.array_equal(superpixels[j+1][i], current_color):
                superpixels[j][i] = current_color
                labels[j+1][i] = current_label
        except IndexError:
                pass # print("BORDER WARNING")
        try:
            if labels[j+1][i+1] > current_label and np.array_equal(superpixels[j+1][i+1], current_color):
                superpixels[j][i] = current_color
                labels[j+1][i+1] = current_label
        except IndexError:
                pass # print("BORDER WARNING")
        try:
            if labels[j+1][i-1] > current_label and np.array_equal(superpixels[j+1][i-1], current_color):
                superpixels[j][i] = current_color
                labels[j+1][i-1] = current_label
        except IndexError:
                pass # print("BORDER WARNING")
        try:
            if labels[j-1][i+1] > current_label and np.array_equal(superpixels[j-1][i+1], current_color):
                superpixels[j][i] = current_color
                labels[j-1][i+1] = current_label
        except IndexError:
                pass # print("BORDER WARNING")
        try:
            if labels[j][i-1] > current_label and np.array_equal(superpixels[j][i-1], current_color):
                labels[j][i-1] = current_label
                superpixels[j][i+1] = current_color
        except IndexError:
                pass # print("BORDER WARNING")
        try:
            if labels[j-1][i] > current_label and np.array_equal(superpixels[j-1][i], current_color):
                superpixels[j][i] = current_color
                labels[j-1][i] = current_label
        except IndexError:
                pass # print("BORDER WARNING")
        try:
            if labels[j-1][i-1] > current_label and np.array_equal(superpixels[j-1][i-1], current_color):
                superpixels[j][i] = current_color
                labels[j-1][i-1] = current_label
        except IndexError:
                pass # print("BORDER WARNING")
superpixels_optimized = color.label2rgb(labels, superpixels, kind='avg')
img_with_branch_2 = segmentation.mark_boundaries(superpixels_optimized, labels)
img_with_branch_1 = segmentation.mark_boundaries(superpixels, labels_optimized)

# np.savetxt('labels.txt', labels, fmt='%d')
np.savetxt('segments_optimized.txt', labels, fmt='%d')
print(len(np.unique(labels.flatten())))

cv2.imshow('superpixels', superpixels)
cv2.imshow('quantized_img', quantized_img)
cv2.imshow('superpixels_optimized', superpixels_optimized)
cv2.imshow('img', img_with_branch_1)
cv2.imshow('img2', img_with_branch_2)
cv2.waitKey(0)


        
