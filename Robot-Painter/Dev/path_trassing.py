import numpy as np 

points = np.array([
        [19, 0], [18, 1], [18, 2], [17, 2], [16,3], [15,4],[15,5],[14,5], [13,6], [12,6], [11,7],[10,7],[9,7],[8,7],[7,6],[6,6],[5,6],[4,5],[3,5],[2,5],[1,6],[0,7] 
        ])

start = np.array([19,0])
current_pos = start
steps = np.array([
        [1,0], [0,-1], [-1, 0], [0, 1], [1,-1], [-1,-1], [-1,1], [1,1]
        ])

path = []
print(np.any(np.array_equal(points[0:,], np.array([19,0]))))
while True:
    for step in steps:
        tmp = current_pos + step
        print(tmp )
        if tmp in points:
            path.append(tmp)   
            current_pos = tmp
            break
    break






print(path)
