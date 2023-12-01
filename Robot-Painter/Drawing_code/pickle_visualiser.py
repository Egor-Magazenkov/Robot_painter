"""Simple program to plot data from pickle."""
import sys
import pickle
import cv2
import numpy as np

CANVAS_SIZE = 300

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
print(data)
zeros = np.zeros((CANVAS_SIZE+2, CANVAS_SIZE+2,3), dtype=np.uint8)
for i, t in enumerate(data):
    t['points'] = np.array(t['points'])
    t['points'][:, 1] = (CANVAS_SIZE - (t['points'][:, 1]*1000) % CANVAS_SIZE)/1000
    print(t['points'])
    print(f'Trj: {i}/{len(data)}, color: {t["color"]}')
    for point in t['points']:
        point = point*1000
        if t['color'] == 239:
            zeros[int(point[1])][int(point[0])] = (255,255,255)
            #cv2.circle(zeros, (int(point[0]), int(point[1])), 2, (255,255,255), -1)
            #cv2.circle(zeros, (int(point[0]), int(point[1])), int(t['width']), (255,255,255), -1)
        else:
            zeros[int(point[1])][int(point[0])] = COLORS[t['color']]
        #zeros[int(point[1])][int(point[0])] = (255,255,255)
    cv2.imshow('s', zeros)
    cv2.waitKey(0)
#cv2.rectangle(zeros, (0,0), (400,400), (255,255,255))
cv2.imshow('s', zeros)
while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.imwrite('result.png', zeros)
        cv2.destroyAllWindows()
        break

