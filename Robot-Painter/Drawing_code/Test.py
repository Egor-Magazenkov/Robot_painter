import cv2 
import numpy as np
import sys
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage import color
prompt = 'the black sheep sitting in the dark brown wooden boat in the middle of the ocean with orange oval sun, landscape, aivazovsky, oil'
img = cv2.imread("/home/leo/Downloads/stable_diffusion_ex.png")
img = cv2.GaussianBlur(img, (13,13), 0)
img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

Z = img_LAB.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 25.0)
ret,label,center = cv2.kmeans(Z, 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)

#
# np.savetxt('test.txt', label.reshape((img.shape[0], img.shape[1])), fmt='%d')
res = center[label.flatten()]
res2 = res.reshape((img.shape))
# colorImg = cv2.dilate(colorImg, np.ones((15, 15), np.uint8))
# res2 = cv2.cvtColor(res2, cv2.COLOR_Lab2BGR)
# res2[label] = [255,255,255]   

label = label.reshape((img.shape[0], img.shape[1]))
print(label)

res3 = cv2.cvtColor(res2, cv2.COLOR_Lab2BGR)


segments = slic(res2, n_segments=100, sigma=3, compactness=20, convert2lab=True) # a higher value of compactness leads to squared regions, a higher value of sigma leads to rounded delimitations
np.savetxt('test.txt',segments, fmt='%d')
superpixels = color.label2rgb(segments, img, kind='avg')

components = np.ones(segments.shape)
color_ref = {str(center[i]):i for i in range(12)}
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        components[i][j] = color_ref[str(res2[i][j])]
print(components)
components = np.uint8(components)
print(segments)
msk = np.ones((img.shape[0], img.shape[1]))*255
msk = mark_boundaries(msk, np.uint8(components))
print(find_boundaries(components, connectivity=2).astype(np.uint8))

test = np.ones(img.shape)*255
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        test[i][j] = center[components[i][j]]

print(test)
mask = np.ones((img.shape[0], img.shape[1]))*255
mask = mark_boundaries(mask, segments)
cv2.imshow('mask', mask)
cv2.imwrite('branch_mine.jpg', mask)
cv2.imwrite('res.jpg', res3)
# res4 = np.ones((res3.shape[0], res3.shape[1]))*255
# for i in range(len(center)):
#     res583 = cv2.inRange(res2, center[i], center[i])
#     cnts583, hier = cv2.findContours(res583, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#     # print(center)
#     # print(label.tolist())
#     for c in cnts583:
#         if cv2.contourArea(c) < 100:
#             continue
#         cv2.drawContours(res4, c, -1, 0)
# cv2.imshow('ffdf', res4)
# cv2.imwrite('branch_stupid.jpg', cv2.dilate(cv2.Canny(res3, 100, 255), np.ones((3,3))))

# cv2.waitKey(0)


pt=[20, 500]
value=res2[pt[0]][pt[1]]
# print(list(value))
# print(res2[pt[0]][pt[1]])
component = []
used = np.ones((res2.shape[0], res2.shape[1]))
good = np.ones(res2.shape)*255
import itertools
# print(list(itertools.product((-1,0,1), (-1,0,1))))
def dfs(pt):
    global used, component
    print(pt)
    used[pt[0]][pt[1]] = 0
    for i,j in zip([-1,0,0,1], [0,-1,1,0]):
        if i==0 and j==0: continue
        pt_ = [pt[0]+i, pt[1]+j]
        if pt_[0]<0 or pt_[1]<0 or pt_[0] >= res2.shape[0] or pt_[1] >= res2.shape[1]: continue
        # print(res2[pt_[0]][pt_[1]])
        if used[pt_[0]][pt_[1]] == 1 and np.array_equal(res2[pt_[0]][pt_[1]], value):
            # component.append(pt_)
            good[pt_[0]][pt_[1]] = (0,0,0)
            # print(good[good!=(255,255,255)])
            # cv2.imshow('sss',good)
            # cv2.waitKey(1)
            dfs(pt_)

# dfs(pt)
# print('Fuck no')
# cv2.imshow('good', good)
# cv2.imwrite('pic1_result.jpeg', res2)
# res3 = cv2.inRange(res2, res2[0][0], res2[0][0])
# contours, hier = cv2.findContours(res3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# res3 = cv2.drawContours(np.ones(res2.shape), contours, -1, (255,0,0))
# res4 = cv2.inRange(res2, res2[90][0], res2[0][0])
# contours, hier = cv2.findContours(res4, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# res3 = cv2.drawContours(res3, contours, -1, (0,0,255))

# segments = slic(res2, n_segments=200, compactness=0.1, sigma=0, convert2lab=False,start_label=1)
# res3 = np.ones(res2.shape)*255
# res3 = color.label2rgb(segments, res2, kind = 'avg', bg_label = 0)
# res3 = res3.astype(np.uint8)
# img_with_branch_2 = res3#(res2, segments)
# print(img)
# print(res2)
# print(cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY))
# res3 = edged = cv2.Canny(res2, 30, 200)
# res3 = res3.astype(np.uint8)
# contours, hier = cv2.findContours(res3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# for cnt in contours:
#     # if cv2.contourArea(cnt) < 1:
#     #     continue
#     cv2.drawContours(res2, cnt, -1, (255,0,0)) 
#     cv2.imshow('ss', res2)
# cv2.imshow('s', superpixels)

cv2.imshow('res2', res3)
cv2.waitKey(0)

# import matplotlib.pyplot as plt
# # fig, ax = plt.subplots()
# # ax.scatter(a,b)

# # plt.show()
# # print(a)
# # print(a.flatten())
# ab_plane = cv2.merge([a.flatten(),b.flatten()])
# ab_plane = np.array([a[0] for a in ab_plane])
# # print(ab_plane)
# hull = ConvexHull(ab_plane)
# print(hull.points)
# plt.scatter(a,b)
# res=0
# r = []
# result = ab_plane[hull.vertices]


# for simplex in hull.simplices:
#     plt.plot(ab_plane[simplex, 0], ab_plane[simplex, 1], 'k-')
#     # r.append(ab_plane[simplex])
#     # print(ab_plane[simplex])
#     res+=1
# # print(res)

# plt.show()