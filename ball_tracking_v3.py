#!/usr/bin/env python3
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import socket
import json


INITIAL_MASK_LOWER_HSV = (0, 0, 0)
INITIAL_MASK_UPPER_HSV = (179, 255, 20)
INITIAL_MIN_BALL_RADIUS = 10
INITIAL_MAX_FPS = 30
CAMERA_RESOLUTION = (640, 480)
DISPLAY_LINE_LEN = 64
DWIN = 'Tracking'
CWIN = 'Calibrate'
CONFIG_FILE = 'config.json'


class PositionClient:

    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (host, port)

    def sendPoint(self, p):
        self.send(p[0], p[1])

    def send(self, x, y):
        self.sock.sendto(self.encode(x, y), self.address)

    def encode(self, x, y):
        return "{:f},{:f}".format(x, y).encode("UTF-8")


class Tracker:

    def __init__(self, videoFile, displayFlag, calibrateFlag, host, port):
        self.videoFile = videoFile
        self.videoFlag = bool(videoFile)
        self.displayFlag = displayFlag
        self.calibrateFlag = calibrateFlag
        self.posclient = PositionClient(host, port)
        self.loadConfig()
        self.minFrameTime = 1.0 / self.maxFps
        self.line = deque(maxlen=DISPLAY_LINE_LEN)
        self.save = 0

        if self.displayFlag:
            cv2.namedWindow(DWIN)
        if self.calibrateFlag:
            cv2.namedWindow(CWIN)
            cv2.createTrackbar('Low H', CWIN, self.hsvLower[0], 179, self.setLowerH)
            cv2.createTrackbar('Low S', CWIN, self.hsvLower[1], 255, self.setLowerS)
            cv2.createTrackbar('Low V', CWIN, self.hsvLower[2], 255, self.setLowerV)
            cv2.createTrackbar('Up H', CWIN, self.hsvUpper[0], 179, self.setUpperH)
            cv2.createTrackbar('Up S', CWIN, self.hsvUpper[1], 255, self.setUpperS)
            cv2.createTrackbar('Up V', CWIN, self.hsvUpper[2], 255, self.setUpperV)
            cv2.createTrackbar('Min R', CWIN, self.minRadius, 100, self.setMinRadius)
            cv2.createTrackbar('Max FPS (restart required)', CWIN, self.maxFps, 120, self.setMaxFps)
            cv2.createTrackbar('Save', CWIN, 0, 1, self.saveConfigFromUi)

    def run(self):
        if self.videoFlag:
            vs = cv2.VideoCapture(self.videoFile)
        else:
            vs = VideoStream(
                src=0,
                usePiCamera=hasPiCamera(),
                resolution=CAMERA_RESOLUTION,
                framerate=self.maxFps
            ).start()

        crects = (
            (int(CAMERA_RESOLUTION[0] / 2) - 30, CAMERA_RESOLUTION[1] - 30),
            (int(CAMERA_RESOLUTION[0] / 2), CAMERA_RESOLUTION[1]),
            (int(CAMERA_RESOLUTION[0] / 2), CAMERA_RESOLUTION[1] - 30),
            (int(CAMERA_RESOLUTION[0] / 2) + 30, CAMERA_RESOLUTION[1])
        )

        # Allow the camera or video file to warm up.
        time.sleep(2.0)

        try:
            lastTime = 0
            while True:

                # Limit FPS.
                dif = time.time() - lastTime
                if dif < self.minFrameTime:
                    time.sleep(self.minFrameTime - dif)
                    dif = time.time() - lastTime
                lastTime += dif

                frame = vs.read()
                if self.videoFlag:
                    frame = frame[1]
                    if frame is None:
                        break
                    frame = imutils.resize(frame, width=CAMERA_RESOLUTION[0])

                #frame = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.hsvLower, self.hsvUpper)
                #mask = cv2.erode(mask, None, iterations=2)
                #mask = cv2.dilate(mask, None, iterations=2)
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = imutils.grab_contours(contours)

                circle = None
                center = None
                if len(contours) > 0:
                    largest = max(contours, key=cv2.contourArea)
                    circle = cv2.minEnclosingCircle(largest)
                    circle = ((int(circle[0][0]), int(circle[0][1])), int(circle[1]))
                    if circle[1] > self.minRadius:
                        #M = cv2.moments(largest)
                        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        center = circle[0]
                        self.posclient.sendPoint(center)

                if self.displayFlag:
                    display = frame.copy()
                    self.line.appendleft(center)
                    if circle:
                        cv2.circle(display, circle[0], circle[1], (0, 255, 255), 2)
                        #cv2.circle(display, center, 5, (0, 0, 255), -1)
                    for i in range(1, len(self.line)):
                        if self.line[i - 1] is None or self.line[i] is None:
                            continue
                        cv2.line(
                            display,
                            self.line[i - 1],
                            self.line[i],
                            (0, 0, 255),
                            int(np.sqrt(DISPLAY_LINE_LEN / float(i + 1)) * 2.5)
                        )
                    cv2.imshow(DWIN, display)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                if self.calibrateFlag:
                    if self.save > 0:
                        self.save -= 1
                        if self.save == 0:
                            cv2.setTrackbarPos("Save", CWIN, 0)
                    display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    low = cv2.cvtColor(np.uint8([[self.hsvLower]]), cv2.COLOR_HSV2BGR).tolist()
                    up = cv2.cvtColor(np.uint8([[self.hsvUpper]]), cv2.COLOR_HSV2BGR).tolist()
                    cv2.rectangle(display, crects[0], crects[1], low[0][0], -1)
                    cv2.rectangle(display, crects[2], crects[3], up[0][0], -1)
                    cv2.imshow(CWIN, display)

        except KeyboardInterrupt:
            pass

        if self.videoFlag:
            vs.release()
        else:
            vs.stop()

        cv2.destroyAllWindows()

    def loadConfig(self):
        try:
            with open(CONFIG_FILE) as file:
                data = json.load(file)
            self.hsvLower = tuple(int(v) for v in data['hsvLower'])
            self.hsvUpper = tuple(int(v) for v in data['hsvUpper'])
            self.minRadius = int(data['minRadius'])
            self.maxFps = int(data['maxFps'])
        except:
            self.hsvLower = INITIAL_MASK_LOWER_HSV
            self.hsvUpper = INITIAL_MASK_UPPER_HSV
            self.minRadius = INITIAL_MIN_BALL_RADIUS
            self.maxFps = INITIAL_MAX_FPS

    def saveConfig(self):
        try:
            with open(CONFIG_FILE, 'w') as file:
                json.dump({
                    'hsvLower': self.hsvLower,
                    'hsvUpper': self.hsvUpper,
                    'minRadius': self.minRadius,
                    'maxFps': self.maxFps
                }, file)
        except:
            pass

    def saveConfigFromUi(self, ignore):
        self.saveConfig()
        self.save = 10

    def setMaskRange(self, upFlag, key, value):
        if upFlag:
            self.hsvUpper = hsvAdjust(self.hsvUpper, key, value)
        else:
            self.hsvLower = hsvAdjust(self.hsvLower, key, value)

    def setLowerH(self, value):
        self.setMaskRange(False, 'h', value)

    def setLowerS(self, value):
        self.setMaskRange(False, 's', value)

    def setLowerV(self, value):
        self.setMaskRange(False, 'v', value)

    def setUpperH(self, value):
        self.setMaskRange(True, 'h', value)

    def setUpperS(self, value):
        self.setMaskRange(True, 's', value)

    def setUpperV(self, value):
        self.setMaskRange(True, 'v', value)

    def setMinRadius(self, value):
        self.minRadius = value

    def setMaxFps(self, value):
        self.maxFps = value


def hsvAdjust(hsv, key, value):
    if key == 'h':
        return (value, hsv[1], hsv[2])
    if key == 's':
        return (hsv[0], value, hsv[2])
    if key == 'v':
        return (hsv[0], hsv[1], value)
    return hsv


def hasPiCamera():
    try:
        import picamera
        return True
    except:
        return False


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-d", "--display",
    help="display frame including tracking data, default: yes")
ap.add_argument("-c", "--calibrate",
    help="display calibration window, default: no")
ap.add_argument("-o", "--host",
    help="coordinate receiving host", type=str, default="192.168.1.50")
ap.add_argument("-p", "--port",
    help="coordinate receiving port", type=int, default=11000)
args = vars(ap.parse_args())

tracker = Tracker(
    args.get("video"),
    args.get("display") not in ("no", "false", "n", "0"),
    args.get("calibrate") in ("yes", "true", "y", "1"),
    args["host"],
    args["port"]
)
tracker.run()