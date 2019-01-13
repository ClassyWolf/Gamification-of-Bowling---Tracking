# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import socket

class PositionClient:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (host, port)

    def send(self, x, y):
        self.sock.sendto(self.encode(x, y), self.address)

    def encode(self, x, y):
        return "{:f},{:f}".format(x, y).encode("UTF-8")

posclient = PositionClient('192.168.1.50', 11000)

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
ap.add_argument("-d", "--display",
    help="display frame including tracking data, default: yes")
args = vars(ap.parse_args())

videoFlag = bool(args.get("video", ""))
displayFlag = args.get("display") not in ("no", "false", "0")

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
hsvLower = (0, 0, 0)
hsvUpper = (360, 0, 0)
minRadius = 15
maxFps = 30
minFrameTime = 1.0 / maxFps
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not videoFlag:
    vs = VideoStream(
        src=0,
        usePiCamera=True,
        resolution=(640, 480),
        framerate=30
    ).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
 
# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
try:
    lastTime = 0
    while True:
        dif = time.time() - lastTime
        if dif < minFrameTime:
            time.sleep(minFrameTime - dif)
            dif = time.time() - lastTime
        fps = 1.0 / dif
        lastTime += dif
        print("{:3.0f} fps".format(fps), end="\r")
        
        # grab the current frame
        frame = vs.read()
     
        # handle the frame from VideoStream
        if videoFlag:
            frame = frame[1]
     
            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if frame is None:
                break
     
        # resize the frame, blur it, and convert it to the HSV
        # color space
        #frame = imutils.resize(frame, width=600)
        #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, hsvLower, hsvUpper)
        #mask = cv2.erode(mask, None, iterations=2)
        #mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
     
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
     
            # only proceed if the radius meets a minimum size
            if radius > minRadius:
                #M = cv2.moments(c)
                #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                center = (int(x), int(y))
                print("(x {:d}, y {:d})".format(center[0], center[1]))
                
                if displayFlag:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                 
                 
                def mapObjectPosition (x, y):
                    mapObjectPosition(int(x), int(y))
                    print("[INFO] Object coordinates = ".format(x, y))
                    
                   
                posclient.send(x, y)
                    


        if displayFlag:

            # update the points queue
            pts.appendleft(center)

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
         
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
         
            # show the frame to our screen
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
         
            # if the 'q' key is pressed, stop the loop
            if key in (ord("q"), 27):
                break




except KeyboardInterrupt:
    pass

# if we are not using a video file, stop the camera video stream
if not videoFlag:
    vs.stop()

# otherwise, release the camera
else:
    vs.release()
 
# close all windows
cv2.destroyAllWindows()
