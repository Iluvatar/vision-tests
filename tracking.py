import sys

import numpy as np

import cv2

OBJECT_RECOGNITION_THRESHOLD_SIZE = 400


class TrackingObject:
    """class for keeping track of moving objects"""

    nextID = 0

    def __init__(self, tracking_threshold):
        self.tracking_threshold = tracking_threshold

        self.time_seen = 0
        self.consec_time_unseen = -1

        self.center = (0, 0)
        self.box = (0, 0, 0, 0)

        self.path = []
        self.disappearance_indices = []

        self.id = TrackingObject.nextID
        TrackingObject.nextID += 1

    def update(self, contour):
        if contour is not None:
            self.consec_time_unseen = 0
            self.time_seen += 1

            # Update center of gravity
            moments = cv2.moments(contour)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            self.center = (cx, cy)

            self.box = cv2.boundingRect(contour)

            self.path.append(self.center)
        else:
            self.consec_time_unseen += 1
            self.disappearance_indices.append(len(self.path) - 1)

    def highlight_object(self, f, trace=False, stats=False):
        if self.consec_time_unseen == 0:
            other_edge = (self.box[0] + self.box[2], self.box[1] + self.box[3])
            cv2.rectangle(f, self.box[0:2], other_edge, (0, 0, 255), 1)
            cv2.circle(f, self.center, 3, (0, 0, 255), -1)

        if trace:
            f_copy = f.copy()
            for i in range(max(0, len(self.path) - 20), len(self.path) - 1):
                if i not in self.disappearance_indices:
                    cv2.line(f, self.path[i], self.path[i + 1], (0, 0, 0), 2)
                    cv2.addWeighted(f, .8, f_copy, .2, 0, f)

        if stats:
            t = self.tracking_threshold
            if len(self.path) > t:
                subset = self.path[-t:]
                line_end_x = (subset[t - 1][0] - subset[t - 3][0]) * 2 + subset[t - 1][0]
                line_end_y = (subset[t - 1][1] - subset[t - 3][1]) * 2 + subset[t - 1][1]
                cv2.arrowedLine(f, subset[t - 1], (line_end_x, line_end_y), (255, 0, 0), 2)

                prediction = self.predicted_path(5)
                for p in prediction:
                    cv2.circle(f, p, 3, (50, 255, 50), -1)

    def predicted_path(self, n):
        t = self.tracking_threshold
        if len(self.path) >= t:
            new_path = [self.path[i * 3] for i in range(len(self.path) / 3)]
            x_data = [i[0] for i in new_path]
            y_data = [i[1] for i in new_path]
            new_x = self._taylor(x_data, 2, n)
            new_y = self._taylor(y_data, 2, n)
            return zip(new_x, new_y)
        else:
            return self.center

    @staticmethod
    def _taylor(data, degree, n=1):
        degree = min(degree, len(data) - 1)

        data_array = [data]
        for i in range(degree):
            d = []

            for j in range(len(data_array[i]) - 1):
                d.append(data_array[i][j + 1] - data_array[i][j])

            data_array.append(d)

        for i in range(n):
            for j in range(degree - 1, -1, -1):
                data_array[j].append(data_array[j][-1] + data_array[j + 1][-1])

        return data_array[0][-n:]


camera_num = 0

# Get command line arguments
if len(sys.argv) == 2:
    camera_num = int(sys.argv[1])


# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 720

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# FRAME_WIDTH = 360
# FRAME_HEIGHT = 260

FPS = 20
SCALE = 2

cap = cv2.VideoCapture(camera_num)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

track = TrackingObject(10)

while cap.isOpened():

    _, frame = cap.read()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask out everything except neon green
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Blur to remove noise
    kernel = np.ones((3, 3), np.float32) / 9
    mask = cv2.filter2D(mask, -1, kernel)

    # Get a list of the contours around each blob
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get indices of blobs sorted from smallest to largest
        blobs = np.argsort(map(cv2.contourArea, contours))

        # Only consider large blobs
        if cv2.contourArea(contours[blobs[-1]]) > OBJECT_RECOGNITION_THRESHOLD_SIZE:
            track.update(contours[blobs[-1]])
        else:
            track.update(None)

        track.highlight_object(frame, trace=True, stats=True)

    cv2.imshow("window", np.fliplr(frame))

    # Quit on 'q' press
    press = cv2.waitKey(10)
    if press & 0xFF == ord('q'):
        break
