import cv2
import numpy as np
from random import randrange
import math
import os

class TestModule:
    def __init__(self):
        # Keep this code the same
        try:
            import libjevois as jevois
            self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
            self.on_jevois = True
        except:
            self.on_jevois = False
        self.frame = 0
        self.output_dir = None
        self.parameters = ['', '']
        self.width = 320
        self.height = 240

    def create_output_dir(self):
        base_dir = 'data'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        folder_exists = True
        folder_number = 1
        while folder_exists:
            current_dir = os.path.join(base_dir, str(folder_number))
            if not os.path.exists(current_dir):
                folder_exists = False
                os.makedirs(current_dir)
                self.output_dir = current_dir
            else:
                folder_number += 1

    def dist(self, P1, P2):
        x1, y1 = P1
        x2, y2 = P2
        dx = abs(x1-x2)
        dy = abs(y1-y2)
        distance = math.sqrt(dx**2 + dy**2)
        return distance

    def isTargetColour(self, colour):
        lower_orange = np.array([0, 0, 130])  # Adjusted lower bound of orange color in YUV
        upper_orange = np.array([255, 140, 255])  # Adjusted upper bound of orange color in YUV
        if np.all(colour >= lower_orange) and np.all(colour <= upper_orange):
            return True

    def searchUpDown(self, P0, yuv):
        x0, y0 = P0
        x1, y1 = x0, y0
        x2, y2 = x0, y0
        done = False
        while not done:
            target_color = self.isTargetColour(yuv[y1 - 1, x1])
            if target_color:
                y1 -= 1
            elif self.isTargetColour(yuv[y1 - 1, x1 - 1]):
                y1 -= 1
                x1 -= 1
            elif self.isTargetColour(yuv[y1 - 1, x1 + 1]):
                y1 -= 1
                x1 += 1
            else:
                done = True
            if y1 == 0:
                done = True

        done = False
        while not done:
            target_color = self.isTargetColour(yuv[y2 + 1, x2])
            if target_color:
                y2 += 1
            elif self.isTargetColour(yuv[y2 + 1, x2 - 1]):
                y2 += 1
                x2 -= 1
            elif self.isTargetColour(yuv[y2 + 1, x2 + 1]):
                y2 += 1
                x2 += 1
            else:
                done = True
            if y2 == 239:
                done = True

        P1 = (x1, y1)
        P2 = (x2, y2)
        return P1, P2, x1, y1

    def searchLeftRight(self, P0, yuv):
        x0, y0 = P0
        xl, yl = x0, y0
        xr, yr = x0, y0
        done = False
        while not done:
            target_color = self.isTargetColour(yuv[yl, xl - 1])
            if target_color:
                xl -= 1
            elif self.isTargetColour(yuv[yl - 1, xl - 1]):
                yl -= 1
                xl -= 1
            elif self.isTargetColour(yuv[yl + 1, xl - 1]):
                yl += 1
                xl -= 1
            else:
                done = True
            if yl == 239:
                done = True

        done = False
        while not done:
            if yr == 239:
                done = True
            target_color = self.isTargetColour(yuv[yr, xr + 1])
            if target_color:
                xr += 1
            elif self.isTargetColour(yuv[yr - 1, xr + 1]):
                yr -= 1
                xr += 1
            elif self.isTargetColour(yuv[yr + 1, xr + 1]):
                yr += 1
                xr += 1
            else:
                done = True
            if yr == 239:
                done = True

        Pl = (xl, yl)
        Pr = (xr, yr)
        dist_Pl = self.dist(P0, Pl)
        dist_Pr = self.dist(P0, Pr)
        if dist_Pl > dist_Pr:
            PN = Pl
        else:
            PN = Pr
        return PN

    def process(self, inframe, outframe):
        if self.on_jevois:
            bgr = inframe.getCvBGR()
            self.timer.start()
        else:
            bgr = cinframe

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        h, w, d = yuv.shape

        max_samples = 800
        sigmaL = 60

        detectedGates = []
        i = 0
        while i < max_samples and len(detectedGates) == 0:
            x0 = randrange(2, w - 3)
            y0 = randrange(4, h - 3)
            P0 = (x0, y0)
            colour0 = yuv[y0, x0]
            if self.isTargetColour(colour0):
                P1, P2 = self.searchUpDown(P0, yuv)[:2]
                if self.dist(P1, P2) > sigmaL:
                    P3 = self.searchLeftRight(P1, yuv)
                    P4 = self.searchLeftRight(P2, yuv)
                    if self.dist(P1, P3) > sigmaL and self.dist(P2, P4) > sigmaL:
                        detectedGate = [P1, P2, P4, P3]
                        detectedGates.append(detectedGate)
                        cv2.polylines(bgr, [np.array(detectedGate)], isClosed=True, color=(0, 255, 0), thickness=2)
            i += 1



        output_image = bgr.copy()
        # Keep code from here
        # --------------------------------------------------
        if self.on_jevois:
            jevois.sendSerial("Parameters frame {} - {}".format(self.parameters[0], self.parameters[1]));

            if self.output_dir is None:
                self.create_output_dir()

            # Save the processed image
            cv2.imwrite(f'{self.output_dir}/img{self.frame}.png', output_image)
            self.frame += 1

            fps = self.timer.stop()
            height, width, _ = output_image.shape
            cv2.putText(output_image, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            outframe.sendCv(output_image)
        else:
            return output_image, self.parameters



    def processNoUSB(self, inframe, outframe, cinframe):
        # Keep this code
        if self.on_jevois:
            bgr = inframe.getCvBGR()
            self.timer.start()
        else:
            bgr = cinframe
        # Add code from here
        # --------------------------------------------------


        # Convert the image to YUV color space
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        # Define the lower and upper bounds of the orange color in YUV
        lower_orange = np.array([0, 0, 160])  # Lower bound of orange color in YUV
        upper_orange = np.array([255, 120, 255])  # Upper bound of orange color in YUV

        # Threshold the image to get a binary mask of the orange color
        mask = cv2.inRange(yuv, lower_orange, upper_orange)

        # Find contours of the orange in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image for drawing
        output_image = bgr.copy()

        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw bounding rectangle around the largest contour and add centroid dot
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw a dot at the centroid
                cv2.circle(output_image, (cX, cY), 3, (0, 0, 255), -1)



        # Keep code from here
        # ----------------------------------------------------------
        if not self.on_jevois:
            return output_image, self.parameters
        else:
            jevois.sendSerial("Parameters frame {} - {}".format(self.parameters[0], self.parameters[1]));

            if self.output_dir is None:
                self.create_output_dir()
            # Save the processed image
            cv2.imwrite(f'{self.output_dir}/img{self.frame}.png', output_image)
            self.frame += 1

            fps = self.timer.stop()
            height, width, _ = output_image.shape
            cv2.putText(output_image, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

