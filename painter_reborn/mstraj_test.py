import numpy as np
import pandas as pd
import os
import traceback
import time
import re
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import URDriver
import median_filter as md
# import roboticstoolbox.tools.trajectory as tr
# import matplotlib.pyplot as plt

orient = np.array([0, 3.14, 0])
position = np.array([0.0346, -0.6279, 0.3532])
POSE = np.concatenate((position, orient))
HOME_POSITION = np.array([-90, -90, -90, 0, 90, 0]) * np.pi / 180
CANVAS_HEIGHT = 0.0
HEIGHT_ERROR = 0.007
TOUCH_CHECK = False
flag = 0

def find_object_force_control(force: np.ndarray, z_force) -> np.ndarray:
    global STAGE
    global TIME_TOUCH
    global last_err
    global integral
    global integral_time
    global forceArray
    global flag
    global START_FORCE
    global CANVAS_HEIGHT
    global HEIGHT_ERROR
    # Coefficients
    max_speed = 0.07

    vel = np.zeros(6)
    k_p = -0.005
    k_i = -0.00001
    k_d = -0.00001

    # time
    i_new_time = time.time()
    delta_t = i_new_time - integral_time
    integral_time = i_new_time

    f_6 = force[:3]

    data = np.zeros(6)
    data[:3] = force[:3]

    # if flag == 0:
    #     flag = 1
    #     START_FORCE = 0.0

    # f_6[2] -= START_FORCE

    # error
    goal = z_force
    err_f = goal - f_6[2]
    derr_f = (err_f - last_err)/delta_t
    last_err = err_f
    # print("Force: ", f_6[2])
    #drawing graph
    #forceArray.append(f_6[2])
    # update_graph(f_6[2])

    # PID
    sum_f = k_p*err_f
    sum_f = sum_f if abs(sum_f) < max_speed else max_speed * np.sign(float(sum_f))
    sum_d = k_d*derr_f
    integral += k_i* delta_t * err_f

    dirv = -DIR_F if abs(f_6[2]) < 10 else DIR_F*0

    f_6[0] = 0
    f_6[1] = 0
    f_6[2] = sum_f + sum_d + dirv[2]

    # print(f_6[2])
    vel[:3] = f_6
    # print('err', err_f, derr_f)
    # robot1.update_state()
    # current_pose = robot1.receive.getActualTCPPose()
    # z = current_pose[2]
    # if np.abs(err_f) < 0.1 and abs(z - CANVAS_HEIGHT) < HEIGHT_ERROR:
    #     STAGE = 1
    # if np.abs(err_f) < 0.1 and CANVAS_HEIGHT == 0.0:
    #     STAGE = 1
    # if (abs(z - CANVAS_HEIGHT) >= 0.02) and (vel[2] > 0) and CANVAS_HEIGHT != 0.0:
    #     vel[2] = -vel[2] 
    if np.abs(err_f) < 0.1:
        STAGE = 1

    return vel




def make_trajectory_speed(data, KP):

    global STAGE
    global TIME_TOUCH
    global count
    global dt
    global speed
    global acceleration
    global orient
    global flag

    up_height = 0.01
    STAGE = 0
    robot1.update_state()
    current_pose = robot1.receive.getActualTCPPose()
    z = current_pose[2]
    x = data[0][0] + canvas_position[0]
    y = data[0][1] + canvas_position[1]
    start_pose = np.array([x, y, z])
    start_pose = np.concatenate((start_pose, orient))
    robot1.control.moveL(start_pose, speed, acceleration, False)

    flag = 0
    while STAGE == 0:
        start = time.time()
        robot1.update_state()
        fe = np.array(robot1.state.f).flatten()
        # flag = 0
        speed_vector = find_object_force_control(fe, 3)
        robot1.control.speedL(speed_vector, acceleration, dt)
        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)



    robot1.control.speedStop()
    # start_pose = np.array([x, y, z - up_height])
    # start_pose = np.concatenate((start_pose, orient))
    # robot1.control.moveL(start_pose, speed, acceleration, False)

    robot1.update_state()
    current_pose = robot1.receive.getActualTCPPose()
    z = current_pose[2]

    last_q = data[0]
    # print(last_q)
    with open('output.csv', 'w') as file:
        file.write("TIME X_GOAL Y_GOAL X_REAL Y_REAL VELOCITY_X VELOCITY_Y")
        file.write('\n')
        for q in data[1:]:
            start = time.time()
            robot1.update_state()
            current_pose = robot1.receive.getActualTCPPose()
            fe = np.array(robot1.state.f).flatten()
            speed_vector = find_object_force_control(fe, 2)
            vx = (q[0] - last_q[0])/dt
            vy = (q[1] - last_q[1])/dt
            speed_vector[0] = vx
            speed_vector[1] = vy



            last_q[0] +=  canvas_position[0]
            last_q[1] +=  canvas_position[1]
            pos_error = last_q[0:2] - current_pose[0:2]
            pos_error = np.concatenate((pos_error, np.array([0])))
            print(pos_error)
            speed_vector[0] = speed_vector[0] + KP * pos_error[0]
            speed_vector[1] = speed_vector[1] + KP * pos_error[1]
            # file.write(str(start) + ',' + str(last_q[0]) + ',' + str(last_q[1]) + ',' + str(current_pose[0]) + ',' + str(current_pose[1]) + ',' + str(speed_vector[0]) + ',' + str(speed_vector[1]) + '\n')
            # x = q[0]+ canvas_position[0]
            # y = q[1]+ canvas_position[1]
            # pose = [x, y, z, rx, ry, rz]
            # pose = np.array([x, y, z])
            # pose = np.concatenate((pose, orient))
            robot1.control.speedL(speed_vector, acceleration, dt)
            # robot1.control.servoL(pose, 0.01, 0.5, dt, 0.1, 300)
            last_q = q
            end = time.time()
            duration = end - start
            if duration < dt:
                time.sleep(dt - duration)
    # robot1.control.servoStop(1)
    # time.sleep(1)
    # robot1.control.servoStop()
    robot1.control.speedStop()

    robot1.update_state()
    current_pose = robot1.receive.getActualTCPPose()
    z = current_pose[2] + up_height
    x = current_pose[0]
    y = current_pose[1]
    pose = np.array([x, y, z])
    last_pose = np.concatenate((pose, orient))

    robot1.control.moveL(last_pose, speed, acceleration, False)


def reset_ft_sensor():
    time.sleep(1)
    #for i in range(15):
    robot1.control.zeroFtSensor()
    time.sleep(1)


def make_trajectory(data):
    global TOUCH_CHECK
    global STAGE
    global TIME_TOUCH
    global count
    global dt
    global speed
    global acceleration
    global orient
    global CANVAS_HEIGHT
    global flag
    up_height = 0.02
    STAGE = 0
    robot1.update_state()
    current_pose = robot1.receive.getActualTCPPose()
    z = current_pose[2]
    x = data[0][0] + canvas_position[0]
    y = data[0][1] + canvas_position[1]
    # print(x, y)
    start_pose = np.array([x, y, z])
    start_pose = np.concatenate((start_pose, orient))
    print('start_pose', start_pose)
    robot1.control.moveL(start_pose, speed, acceleration, False)
    
    while STAGE == 0:
        start = time.time()
        robot1.update_state()
        fe = np.array(robot1.state.f).flatten()
        speed_vector = find_object_force_control(fe, 2)
        robot1.control.speedL(speed_vector, acceleration, dt)
        end = time.time()
        duration = end - start
        
        if duration < dt:
            time.sleep(dt - duration)
    
    if TOUCH_CHECK == False:
        TOUCH_CHECK == True
        robot1.update_state()
        current_pose = robot1.receive.getActualTCPPose()
        CANVAS_HEIGHT = current_pose[2]

    robot1.control.speedStop()


    robot1.update_state()
    current_pose = robot1.receive.getActualTCPPose()
    z = current_pose[2]

    last_q = data[0]
    # print(last_q)
    for q in data:
        start = time.time()
        robot1.update_state()
        current_pose = robot1.receive.getActualTCPPose()

        x = q[0] + canvas_position[0]
        y = q[1] + canvas_position[1]
        # print(x, y)
        # pose = [x, y, z, rx, ry, rz]
        pose = np.array([x, y, z])
        pose = np.concatenate((pose, orient))
        # robot1.control.speedL(speed_vector, acceleration, dt)
        robot1.control.servoL(pose, 0.01, 0.5, dt, 0.1, 300)
        last_q = q
        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)

    robot1.control.servoStop()

    robot1.update_state()
    current_pose = robot1.receive.getActualTCPPose()
    
    z = current_pose[2] + up_height
    x = current_pose[0]
    y = current_pose[1]
    pose = np.array([x, y, z])
    last_pose = np.concatenate((pose, orient))
    # flag = 0
    robot1.control.moveL(last_pose, speed, acceleration, False)


def draw_canvas_edges():
    global STAGE
    STAGE = 0

    canvas_pose = np.concatenate((canvas_position, orient))
    speed = 1
    acceleration = 0.5
    blend = 0

    robot1.control.moveL(canvas_pose, speed, acceleration)

    while STAGE == 0:
        start = time.time()
        robot1.update_state()
        fe = np.array(robot1.state.f).flatten()
        speed_vector = find_object_force_control(fe, 3)
        robot1.control.speedL(speed_vector, acceleration, dt)
        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)

    robot1.control.speedStop()
    robot1.update_state()
    current_pose = robot1.receive.getActualTCPPose()
    z = current_pose[2]
    canvas_pose[2] = z

    canvas_pose[0] += 0.4
    robot1.control.moveL(canvas_pose, speed, acceleration)
    canvas_pose[1] += 0.4
    robot1.control.moveL(canvas_pose, speed, acceleration)
    canvas_pose[0] -= 0.4
    robot1.control.moveL(canvas_pose, speed, acceleration)
    canvas_pose[1] -= 0.4
    robot1.control.moveL(canvas_pose, speed, acceleration)
    canvas_pose[2] += up_height
    robot1.control.moveL(canvas_pose, speed, acceleration)



def drawingV01(directory):

    # Main part
    files = os.listdir(directory)

    for i in range(0, len(files)):
        try:
            filename = os.path.join(directory, files[i])
            data = pd.read_csv(filename).to_numpy()
            print("Trj: ", i)
            make_trajectory(data)
            # make_trajectory_speed(data, 0)
            # make_trajectory_speed(data, 8)
        except Exception:
            traceback.print_exc()


def drawingV02(data):
	
    speed = 3	
    n = 0
    trajectories = data['trajectories'][n:]
    color_num = trajectories[0]['color']
    trj_num = len(data['trajectories'])
    # color_num = 0
    for i, t in enumerate(trajectories[290:]):
	
        color_trj = t['color']
        robot1.control.zeroFtSensor()
        # time.sleep(100)
        # color_trj = color_num
        if (color_trj <= 0):
            continue
        points = t['points']
        print(points)
        if (abs(color_num - color_trj) > 0):
            
            robot1.control.moveL(POSE, speed, acceleration)
            robot1.control.servoStop()
            robot1.control.moveJ(HOME_POSITION)
            
            input1 = input('Press enter')
            color_num = color_trj
            robot1.control.moveL(POSE, speed, acceleration)

        print('[i / truajectories]: {}/{}'.format(i + n, trj_num))
        make_trajectory(points)


if __name__ == '__main__':
    global STAGE
    # Setup robot
    ip = '192.168.88.6'
    robot1 = URDriver.UniversalRobot(ip)

    DIR_F   = np.array([0, 0, -0.1, 0, 0, 0])*0.01
    NUM_JOINTS = 6
    # Joint movement
    home_position = np.array([-90, -90, -90, 0, 90, 0]) * np.pi / 180
    robot1.control.moveJ(home_position)


    # Cartesian space movements
    up_height = 0.01
    canvas_position = np.array([-0.212, -0.460 - 0.4, 0.2])

    orient = np.array([0, 3.14, 0])
    position = np.array([0.0346, -0.6279, 0.3])
    pose = np.concatenate((position, orient))
    canvas_pose = np.concatenate((canvas_position, orient))
    # Movement configuration
    speed = 3
    acceleration = 1
    blend = 0

    robot1.control.moveL(pose, speed, acceleration)
    #draw_canvas_edges()
    # variables for PID control
    last_err = 0
    integral = 0
    integral_time = 0

    dt = 1.0/500

    median = md.MedianFilter(NUM_JOINTS, 5)

    # path's length in meters
    R = 0.08
    length = 2*np.pi*R
    Vmax = 0.01
    a2 = 8*Vmax*Vmax/(3*length)
    a1 = -np.sqrt((a2**3)/(6*length))
    # print(a1, a2)
    STAGE = 0
    count = 0

    forceArray = []
    time_array = np.array([])
    TIME_TOUCH = 0


    data = np.zeros(6000)
    ptr = 0

    flag = 0
    START_FORCE = 0

    # draw_canvas_edges()

    # directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trj')
    # drawingV01(directory)
    reset_ft_sensor()
    directory = '/home/leo/Downloads/'
    filename = directory + 'trjs_bridge.pickle'
    STAGE = 0
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        print(len(data))
        #draw_canvas_edges()
        drawingV02(data)


