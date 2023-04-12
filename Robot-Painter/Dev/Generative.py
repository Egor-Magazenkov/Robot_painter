import cv2
import numpy as np
from skimage.color import deltaE_ciede2000

def clustering(img, colors=12):
    img_blured = img #cv2.GaussianBlur(img, (9,9), 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img_LAB = cv2.cvtColor(img_blured, cv2.COLOR_BGR2Lab)
    Z = img_LAB.reshape((-1,3))
    Z = np.float32(Z)

    ret,label,center=cv2.kmeans(Z, colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    label = label.reshape((img.shape[0], img.shape[1]))
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res3 = cv2.cvtColor(res2, cv2.COLOR_Lab2BGR)

    return res3


def difference(cur_img, img):
    cur_img_LAB = cv2.cvtColor(cur_img, cv2.COLOR_BGR2Lab)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    res = np.abs(deltaE_ciede2000(cur_img_LAB, img_LAB))
    return res

def difference_with_color(img, color):
    color_img = np.ones_like(img)*color
    return difference(img, color_img)

def find_loss(cur_img, img):
    res = difference(cur_img, img)
    
    return np.mean(np.abs(res))

def closest_to_point(point, set_of_points):
    return np.argmin(np.array([np.hypot(*(point-p)) for p in set_of_points]))


def generate_stroke(start_point, radius, image, canvas):
    x, y = start_point
    stroke_color = image[y][x]
    print(stroke_color)
    loss = find_loss(image,  canvas)


    canvas_new = canvas.copy()
    loss_new = loss.copy()

    window_radius = radius*4
    
    res = [start_point]
    
    while True:
        print(x,y)
        img_tmp = image.copy()
        cv2.circle(canvas_new, (x,y), radius, stroke_color.tolist(), -1)
        cv2.rectangle(img_tmp, (x-window_radius, y-window_radius), (x+window_radius, y+window_radius), thickness=1, color=(0,0,0))
        cv2.imshow('img', img_tmp) 
        cv2.imshow('canvas',canvas_new)

       
            # window_diff = difference(image[max(x-window_radius, 0):min(x+window_radius+1, image.shape[1]-1), max(y-window_radius, 0):min(y+window_radius+1, image.shape[0]-1)], \
            #                         canvas_new[max(x-window_radius, 0):min(x+window_radius+1, image.shape[1]-1), max(y-window_radius, 0):min(y+window_radius+1, image.shape[0]-1)])

        window_diff = difference_with_color(image[max(x-window_radius, 0):min(x+window_radius+1, image.shape[1]-1), max(y-window_radius, 0):min(y+window_radius+1, image.shape[0]-1)], \
                                    stroke_color)
        window_diff *= (1-difference(image[max(x-window_radius, 0):min(x+window_radius+1, image.shape[1]-1), max(y-window_radius, 0):min(y+window_radius+1, image.shape[0]-1)], \
                                    canvas_new[max(x-window_radius, 0):min(x+window_radius+1, image.shape[1]-1), max(y-window_radius, 0):min(y+window_radius+1, image.shape[0]-1)])/255/3)
        window_diff[window_diff.shape[1]//2][window_diff.shape[0]//2] = +np.infty
        print(window_diff)
        while True:
            min_diff_coords = np.where(window_diff==np.min(window_diff))
            # print(min_diff_coords)
            min_diff_coords = np.array([[i, j] for i,j in zip(min_diff_coords[0], min_diff_coords[1])]) 
            min_diff_coords_ = min_diff_coords +np.array([x,y])-np.array([window_diff.shape[0]//2, window_diff.shape[1]//2])
            # print(min_diff_coords_)
            dx, dy = min_diff_coords[closest_to_point((x,y), min_diff_coords_)]
            # dy, dx = np.unravel_index(window_diff.argmin (), window_diff.shape)
            print(dx, dy)
            x_new, y_new = x + dx - window_diff.shape[1]//2, y+dy-window_diff.shape[0]//2
            
            if (x_new, y_new) in res:
                window_diff[dx,dy] = +np.infty
            else:
                res.append((x_new, y_new))
                break
            
       
        x, y = x_new, y_new


        cv2.waitKey(1)
        # while cv2.waitKey(1) != ord(' '): pass
        

if __name__ == '__main__':
    img = cv2.imread('/home/leo/Downloads/pic1.jpeg')
    img_quantized = clustering(img)
    # cv2.imshow('ig', img_quantized)
    canvas = np.ones_like(img)*255
    # print(canvas)

    

    # res = find_loss(img, img_quantized)
    # print(res)
    generate_stroke((200,200), 4, img_quantized, canvas)
    # print(difference(img, img))

    # while cv2.waitKey(0) != ord('q'): pass
