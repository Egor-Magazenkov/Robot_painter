import numpy as np
import cv2
import cv2.aruco

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    makrkersize=4
    totalMarkers=50
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    arucoParam = cv2.aruco.DetectorParameters()
    
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)


    corners, ids, rejected = detector.detectMarkers(img)

    mtx = np.array([[974.48680564,   0.0,         339.14893778],
                    [ 0.0, 975.32274337, 246.35277935],
                    [  0.0,        0.0,          1.0      ]])
    
    dist= np.array([ 0.1286079,  -0.29245357,  0.00269586,  0.00114078,  0.32942485])
    tvec = None
    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.075, mtx, dist)
            print(tvec, rvec)
            cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, 0.2)


    cv2.imshow("test", img)
    cv2.waitKey(1)
