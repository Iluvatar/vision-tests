import time
import math
import numpy as np
import cv2
import sys
import thread
import collections


# face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12/share/OpenCV/haarcascades/haarcascade_eye.xml')



# line = [x, y, dx, dy]
def drawLine(image, line, color, width):
    global oldLineBounds
    refreshPosThresh = 5
    refreshAngThresh = math.pi / 60

    [x, y, dx, dy] = line
    lefty = int(x * -dy / dx + y)
    righty = int((FRAME_WIDTH - x) * dy / dx + y)

    if abs(lefty - oldLineBounds[0] + righty - oldLineBounds[1]) < refreshPosThresh and abs(math.atan2(dy, dx) - oldLineBounds[2]) < refreshAngThresh:
        cv2.line(image, (0, oldLineBounds[0]), (FRAME_WIDTH, oldLineBounds[1]), color, width)
    else:
        oldLineBounds = [lefty, righty, math.atan2(dy, dx)]
        cv2.line(image, (0, lefty), (FRAME_WIDTH, righty), color, width)


# line = [[x1, y1], [x2, y2]] numpy array
# point = [x, y] numpy array
# first return is distance, second is vector from line to point
def getMinDist(line, point):
    [p1, p2] = line
    dist = p2 - p1
    normSquared = np.dot(dist, dist)
    t = np.dot(point - p1, dist) / normSquared
    proj = p1 + t * dist
    return [np.linalg.norm(point - proj), point - proj]


# line = [[x1, y1], [x2, y2]] numpy array
def testClip(line):
    global ball_vel, ball_pos, oldBallPos
    [p1, p2] = line
    mirrorVec = p2 - p1
    [dist, vec] = getMinDist(line, ball_pos)

    # if dist <= BALL_RADIUS or intersect(line, [ball_pos, oldBallPos]):
    if dist <= BALL_RADIUS:
        vel_rej = ball_vel - np.dot(ball_vel, mirrorVec) * mirrorVec / np.dot(mirrorVec, mirrorVec)
        ball_vel -= vel_rej * 2

        rad_vec = vec / dist * BALL_RADIUS
        adjust = rad_vec - vec
        ball_pos += 2 * adjust


# line = [[x1, y1], [x2, y2]] numpy array
def intersect(line1, line2):
    [a, b] = line1
    [c, d] = line2

    cSign = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    dSign = (b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])

    if cSign * dSign > 0:
        return False
    else:
        print "int"
        return True

def display_frame():
    while True:
        try:
            i = frame_buffer.popleft()
        except IndexError as e:
            time.sleep(.02)
        else:
            cv2.imshow("window", np.fliplr(i))










# Get command line arguments
if len(sys.argv) <= 1:
    cameraNum = 0
elif len(sys.argv) == 2:
    cameraNum = int(sys.argv[1])

FRAME_WIDTH = 720 #960 / 2
FRAME_HEIGHT = 405 #540 / 2
FPS = 20
SCALE = 2

frame_buffer = collections.deque()

c = 0
fps_history = 10 * [0]


BALL_RADIUS = 30
    
# Create video capture object
cap = cv2.VideoCapture(cameraNum)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

ball_pos = np.array([FRAME_WIDTH / 2, FRAME_HEIGHT / 4], dtype=np.float)
ball_vel = np.array([25, 0], dtype=np.float)
ball_acc = np.array([0, 1], dtype=np.float)


line = np.array([0, 0, 0, 1], dtype=np.float)
bounds = np.array([[0, 0, 1, 0], [0, FRAME_HEIGHT, 1, 0], [0, 0, 0, 1], [FRAME_WIDTH, 0, 0, 1]], dtype=np.float)
oldLineBounds = [0, 0, 0]
oldBallPos = np.array([FRAME_WIDTH / 2, FRAME_HEIGHT / 4], dtype=np.float)

try:
   thread.start_new_thread(display_frame, () )
except:
   print "Error: unable to start thread"

while (cap.isOpened()):
    start = time.time()
    
    # Capture frame-by-frame
    _, frame = cap.read()




    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     # for (ex,ey,ew,eh) in eyes:
    #     #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)




    # frame = cv2.resize(frame_hd, (FRAME_WIDTH / SCALE, FRAME_HEIGHT / SCALE), interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([60, 60, 0])
    # upper_green = np.array([90, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.float32) / 9
    dst = cv2.filter2D(mask, -1, kernel)

    _, contours, _ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    if len(contours) > 0:
        blobs = np.argsort(map(cv2.contourArea, contours))

        if cv2.contourArea(contours[blobs[-1]]) > 100:
            [dx, dy, x, y] = cv2.fitLine(contours[blobs[-1]], cv2.DIST_L2, 0, 0.01, 0.01)

            drawLine(frame, [x, y, dx, dy], (255, 0, 0), 2)
            line = np.array([x, y, dx, dy], dtype=np.float)
            line = np.ndarray.flatten(line)
        else:
            line = np.array([0, FRAME_HEIGHT * 2, 1, 0], dtype=np.float)
    else:
        line = np.array([0, FRAME_HEIGHT * 2, 1, 0], dtype=np.float)

    





    cv2.circle(frame, tuple(map(int, tuple(ball_pos))), BALL_RADIUS, (0, 0, 255), -1)
    



    # Display  frame
    # frame = cv2.resize(frame, (FRAME_WIDTH / SCALE, FRAME_HEIGHT / SCALE), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("window", np.fliplr(frame))


    frame_buffer.append(frame)



    # update ball pos and vel
    ball_pos += ball_vel


    testClip(np.array([[line[0], line[1]], [line[0] + line[2], line[1] + line[3]]]))

    oldBallPos = ball_pos


    if (ball_pos[0] - BALL_RADIUS < 0 or ball_pos[0] + BALL_RADIUS >= FRAME_WIDTH):
        ball_vel *= (-1, 1)
    # if (ball_pos[1] - BALL_RADIUS < 0 or ball_pos[1] + BALL_RADIUS >= FRAME_HEIGHT):
    if (ball_pos[1] + BALL_RADIUS >= FRAME_HEIGHT):
        ball_vel *= (1, -1)


    if (ball_pos[0] - BALL_RADIUS < 0):
        ball_pos += (2 * (BALL_RADIUS - ball_pos[0]), 0)
    # if (ball_pos[1] - BALL_RADIUS < 0):
    #     ball_pos += (0, BALL_RADIUS - ball_pos[1])

    if (ball_pos[0] + BALL_RADIUS >= FRAME_WIDTH):
        offset = 2 * (FRAME_WIDTH - (ball_pos[0] + BALL_RADIUS))
        ball_pos += (offset, 0)
    if (ball_pos[1] + BALL_RADIUS >= FRAME_HEIGHT):
        offset = 2 * (FRAME_HEIGHT - (ball_pos[1] + BALL_RADIUS))
        ball_pos += (0, offset)


    for b in bounds:
        testClip(np.array([[b[0], b[1]], [b[0] + b[2], b[1] + b[3]]]))


    ball_vel += ball_acc

    # time.sleep(.02)

    
    # Quit on 'q' press
    press = cv2.waitKey(10);
    if press & 0xFF == ord('q'):
        break

    elapsed = time.time() - start
    # if elapsed < 1. / FPS:
    #     time.sleep(1. / FPS - elapsed)
    # elapsed = time.time() - start

    fps_history[c%10] = 1. / elapsed
    
    if not (c%10):
        avg = 0
        for i in range(10): avg += fps_history[i]
        avg /= 10
        print int(round(avg)), "fps"
    c += 1

# Release everything
cap.release()
cv2.destroyAllWindows()










