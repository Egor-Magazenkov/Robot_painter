"""Simple program to plot data from pickle."""
import sys
import pickle
import cv2
import numpy as np

if len(sys.argv) != 3:
    print('Bad argument usage.\n'
               'Usage: python3 pickle_visualiser.py '
                            '<path/to/colors.txt> '
                            '<path/to/contour_trjs.pickle> ')
    sys.exit(1)
else:
    COLORS = np.loadtxt(sys.argv[1])
    PICKLE_FILE = sys.argv[2]

with open(PICKLE_FILE, 'rb') as pickle_file:
    d = pickle.load(pickle_file)
data = d['trajectories']
zeros = np.zeros((400,400,3), dtype=np.uint8)
for t in data:
    for point in t['points']:
        point = point*1000
        cv2.circle(zeros, (int(point[0]), int(point[1])), t['width'], t['color'], -1)
        #zeros[int(point[1])][int(point[0])] = COLORS[t['color']]
        #zeros[int(point[1])][int(point[0])] = (255,255,255)
    cv2.imshow('s', zeros)
    cv2.waitKey(1)
cv2.waitKey(0)
cv2.imwrite('result.png', zeros)
