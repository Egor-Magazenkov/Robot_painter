import pickle
import cv2
import numpy as np

COLORS = np.loadtxt('/home/leo/Documents/Robot_painter/colors.txt')
d = pickle.load(open('/home/leo/Documents/Robot_painter/contour_trjs.pickle', 'rb'))
# d = pickle.load(open('/home/leo/Documents/Lab/Generative_Painterly/PyPainterly/trjs.pickle', 'rb'))
data = d['trajectories']
print(data)
zeros = np.zeros((400,400,3), dtype=np.uint8)
for t in data:
    for point in t['points']:
        point = point*1000
        #zeros[int(point[1])][int(point[0])] = COLORS[t['color']]
        zeros[int(point[1])][int(point[0])] = (255,255,255)
    cv2.imshow('s', zeros)
    cv2.waitKey(1)
cv2.waitKey(0)
