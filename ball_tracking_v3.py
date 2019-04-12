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


CONFIG_FILE = 'config.json'
RESOLUTION = (320, 240)
DISPLAY_LINE_LEN = 32
DWIN = 'Tracking'
CWIN = 'Calibrate'


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


class HSVColor:

    @staticmethod
    def read(src, h, s, v):
        if src and len(src) == 3:
            return HSVColor(*(int(v) for v in src))
        return HSVColor(h, s, v)

    def __init__(self, h, s, v):
        self.vals = (h, s, v)

    @property
    def h(self):
        return self.vals[0]

    @property
    def s(self):
        return self.vals[1]

    @property
    def v(self):
        return self.vals[2]

    def setH(self, value):
        return HSVColor(value, self.vals[1], self.vals[2])

    def setS(self, value):
        return HSVColor(self.vals[0], value, self.vals[2])

    def setV(self, value):
        return HSVColor(self.vals[0], self.vals[1], value)

    def hsv(self):
        return self.vals

    def bgr(self):
        l = cv2.cvtColor(np.uint8([[self.vals]]), cv2.COLOR_HSV2BGR).tolist()
        return l[0][0]


class Quadrangle:

    @staticmethod
    def read(src, tl, tr, br, bl):
        mx = max(p[0] for p in (tl, tr, br, bl))
        my = max(p[1] for p in (tl, tr, br, bl))
        def readPoint(psrc, p):
            if psrc and len(psrc) == 2:
                return (max((0, min((mx, psrc[0])))), max((0, min((my, psrc[1])))))
            return p
        if src and len(src) == 4:
            return Quadrangle(
                readPoint(src[0], tl),
                readPoint(src[1], tr),
                readPoint(src[2], br),
                readPoint(src[3], bl)
            )
        return Quadrangle(tl, tr, br, bl)

    def __init__(self, topLeft, topRight, bottomRight, bottomLeft):
        self.tl = topLeft
        self.tr = topRight
        self.br = bottomRight
        self.bl = bottomLeft

    def moveCorner(self, point):
        center = self.center()
        if point[0] < center[0]:
            if point[1] < center[1]:
                self.tl = point
            else:
                self.bl = point
        else:
            if point[1] < center[1]:
                self.tr = point
            else:
                self.br = point

    def center(self):
        return (
            sum(p[0] for p in self.points()) / 4,
            sum(p[1] for p in self.points()) / 4
        )

    def points(self):
        return (self.tl, self.tr, self.br, self.bl)

    def lines(self, image, color):
        cv2.polylines(
            image,
            [np.array(self.points(), dtype='int32')],
            True,
            color,
            1
        )

    def fill(self, image, color):
        cv2.fillPoly(
            image,
            [np.array(self.points(), dtype='int32')],
            color
        )


class Tracker:

    def __init__(self, displayFlag, calibrateFlag, host, port):
        self.piCameraFlag = hasPiCamera()
        self.displayFlag = displayFlag or calibrateFlag
        self.calibrateFlag = calibrateFlag
        self.posclient = PositionClient(host, port)
        self.loadConfig(CONFIG_FILE, RESOLUTION)
        self.minFrameTime = 1.0 / self.maxFps
        self.line = deque(maxlen=DISPLAY_LINE_LEN)
        self.save = 0

        if self.displayFlag:
            print('Press ESC or Q to exit.')
            cv2.namedWindow(DWIN)
        if self.calibrateFlag:
            cv2.namedWindow(CWIN)
            cv2.createTrackbar('Low H', CWIN, self.hsvLower.h, 179, self.setLowerH)
            cv2.createTrackbar('Low S', CWIN, self.hsvLower.s, 255, self.setLowerS)
            cv2.createTrackbar('Low V', CWIN, self.hsvLower.v, 255, self.setLowerV)
            cv2.createTrackbar('Up H', CWIN, self.hsvUpper.h, 179, self.setUpperH)
            cv2.createTrackbar('Up S', CWIN, self.hsvUpper.s, 255, self.setUpperS)
            cv2.createTrackbar('Up V', CWIN, self.hsvUpper.v, 255, self.setUpperV)
            cv2.createTrackbar('Min R', CWIN, self.minRadius, 100, self.setMinRadius)
            cv2.createTrackbar('Max R', CWIN, self.maxRadius, 100, self.setMaxRadius)
            cv2.createTrackbar('Max FPS (restart required)', CWIN, self.maxFps, 120, self.setMaxFps)
            cv2.createTrackbar('Save', CWIN, 0, 1, self.saveConfigFromUi)
            cv2.setMouseCallback(DWIN, self.setCorner)

    def run(self):
        vs = VideoStream(
            src=0,
            usePiCamera=self.piCameraFlag,
            resolution=RESOLUTION,
            framerate=self.maxFps
        ).start()

        midx = int(RESOLUTION[0] / 2)
        colorRects = (
            (midx - 30, RESOLUTION[1] - 30),
            (midx, RESOLUTION[1]),
            (midx, RESOLUTION[1] - 30),
            (midx + 30, RESOLUTION[1])
        )

        # Allow the camera to warm up
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
                if not self.piCameraFlag:
                    frame = cv2.resize(frame, RESOLUTION)
                #frame = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.hsvLower.hsv(), self.hsvUpper.hsv())
                mask = cv2.bitwise_and(mask, mask, mask=self.cornersMask)
                #mask = cv2.erode(mask, None, iterations=2)
                #mask = cv2.dilate(mask, None, iterations=2)
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = imutils.grab_contours(contours)

                circle = None
                center = None
                for i in range(len(contours)):
                    c = cv2.minEnclosingCircle(contours[i])
                    if (
                        (circle is None or c[1] > circle[1]) and
                        c[1] >= self.minRadius and c[1] <= self.maxRadius
                    ):
                        circle = ((int(c[0][0]), int(c[0][1])), int(c[1]))
                        #M = cv2.moments(contours[i])
                        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        center = circle[0]
                        self.posclient.sendPoint(self.applyTransform(center))

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

                    if self.calibrateFlag:
                        if self.save > 0:
                            self.save -= 1
                            if self.save == 0:
                                cv2.setTrackbarPos("Save", CWIN, 0)
                        gui = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(gui, colorRects[0], colorRects[1], self.hsvLower.bgr(), -1)
                        cv2.rectangle(gui, colorRects[2], colorRects[3], self.hsvUpper.bgr(), -1)
                        cv2.imshow(CWIN, gui)
                        self.corners.lines(display, (0, 255, 0))

                    cv2.imshow(DWIN, display)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

        except KeyboardInterrupt:
            pass

        vs.stop()
        cv2.destroyAllWindows()

    def loadConfig(self, fileName, resolution):
        data = {}
        try:
            with open(fileName) as file:
                data = json.load(file)
        except:
            pass
        self.hsvLower = HSVColor.read(data.get('hsvLower'), 0, 0, 0)
        self.hsvUpper = HSVColor.read(data.get('hsvUpper'), 179, 255, 20)
        self.minRadius = int(data.get('minRadius', 10))
        self.maxRadius = int(data.get('maxRadius', 50))
        self.maxFps = int(data.get('maxFps', 30))
        self.corners = Quadrangle.read(
            data.get('corners'),
            (0, 0),
            (resolution[0] - 1, 0),
            (resolution[0] - 1, resolution[1] - 1),
            (0, resolution[1] - 1)
        )
        self.updateTransform()

    def saveConfig(self, fileName):
        try:
            with open(fileName, 'w') as file:
                json.dump({
                    'hsvLower': self.hsvLower.hsv(),
                    'hsvUpper': self.hsvUpper.hsv(),
                    'minRadius': self.minRadius,
                    'maxRadius': self.maxRadius,
                    'maxFps': self.maxFps,
                    'corners': self.corners.points()
                }, file)
        except:
            pass

    def saveConfigFromUi(self, ignore):
        self.saveConfig(CONFIG_FILE)
        self.save = 10

    def setLowerH(self, value):
        self.hsvLower = self.hsvLower.setH(value)

    def setLowerS(self, value):
        self.hsvLower = self.hsvLower.setS(value)

    def setLowerV(self, value):
        self.hsvLower = self.hsvLower.setV(value)

    def setUpperH(self, value):
        self.hsvUpper = self.hsvUpper.setH(value)

    def setUpperS(self, value):
        self.hsvUpper = self.hsvUpper.setS(value)

    def setUpperV(self, value):
        self.hsvUpper = self.hsvUpper.setV(value)

    def setMinRadius(self, value):
        self.minRadius = value

    def setMaxRadius(self, value):
        self.maxRadius = value

    def setMaxFps(self, value):
        self.maxFps = value

    def setCorner(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONUP:
            self.corners.moveCorner((x, y))
            self.updateTransform()

    def updateTransform(self):
        self.cornersMask = np.zeros((RESOLUTION[1], RESOLUTION[0], 1), np.uint8)
        self.corners.fill(self.cornersMask, (255))
        self.transform = cv2.getPerspectiveTransform(
            np.array(self.corners.points(), dtype='float32'),
            np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype='float32')
        )

    def applyTransform(self, point):
        l = cv2.perspectiveTransform(
            np.array([[point]], dtype='float32'),
            self.transform
        )
        return l[0][0]


def hasPiCamera():
    try:
        import picamera
        return True
    except:
        return False


ap = argparse.ArgumentParser()
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
    args.get("display") not in ("no", "false", "n", "0"),
    args.get("calibrate") in ("yes", "true", "y", "1"),
    args["host"],
    args["port"]
)
tracker.run()
