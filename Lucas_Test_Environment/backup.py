import libjevois as jevois
import cv2
import numpy as np
import random
import string
import os


## Test global shutter
#
# Add some description of your module here.
#
# @author Aga Nowak
#
# @videomapping YUYV 320 240 30 BAYER 320 240 30 TUDelft TestModule
# @email a.nowak@tudelft.nl
# @address 123 first street, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by Aga Nowak
# @mainurl tudelft.nl
# @supporturl tudelft.nl
# @otherurl tudelft.nl
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class TestModule:
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
        img = inframe.getCvBGR()

        if False:
            cv2.imwrite(f'/jevois/data/{self.dir}/usb{self.frame}.png', img)
            self.frame += 1

        # Convert our output image to video output format and send to host over USB:
        outframe.sendCv(img)

    # ###################################################################################################
    ## Process function without USB output
    def processNoUSB(self, inframe):
        img = inframe.getCvBGR()
        cv2.imwrite(f'/jevois/data/{self.dir}/img{self.frame}.png', img)
        self.frame += 1
