import libjevois as jevois
import cv2
import numpy as np
import random
import string
import os


## sdlkdj
#
# Add some description of your module here.
#
# @author sdlk
# 
# @videomapping YUYV 320 240 30 YUYV 320 240 30 SKL LucasAndRubenTest
# @email sdklfj@glkd
# @address 123 first street, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by sdlk
# @mainurl dlskd
# @supporturl dlskd
# @otherurl dlskd
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class LucasAndRubenTest:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        self.frame = 0

        # Create a directory with a random name to save images
        self.dir = ''.join(random.choices(string.ascii_lowercase, k=10))
        os.mkdir('data/' + self.dir)

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR. If you need a
        # grayscale image, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB() and getCvRGBA():
        gray = inframe.getCvGRAY()

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # When they do "img = cv2.imread('name.jpg', 0)" in these tutorials, the last 0 means they want a
        # gray image, so you should use getCvGRAY() above in these cases. When they do not specify a final 0 in imread()
        # then usually they assume color and you should use getCvBGR() above.
        #
        # The simplest you could try is:
        #    outimg = inimg
        # which will make a simple copy of the input image to output.

        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.blur(gray, (3, 3))

        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred,
                                            cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                            param2=30, minRadius=1, maxRadius=40)

        # Draw circles that are detected.
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(gray, (a, b), r, (255, 255, 255), 2)
                cv2.circle(gray, (a, b), 1, (0, 0, 255), 3)
        outimg = gray
        if False:
            cv2.imwrite('/jevois/data/{}/img{}.png'.format(self.dir, time.time()), outimg)

        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Convert our output image to video output format and send to host over USB:
        outframe.sendCv(outimg)
        
    # ###################################################################################################
    ## Process function without USB output
    def processNoUSB(self, inframe):
        img = inframe.getCvGRAY()
        cv2.imwrite(f'/jevois/data/{self.dir}/img{self.frame}.png', img)
        self.frame += 1
