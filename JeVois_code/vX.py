import cv2
import numpy as np
from random import randrange
import os
import libjevois as jevois


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
        distance = cv2.norm(P1, P2)
        return distance

    def isTargetColour(self, colour):
        lower_orange = np.array([0, 0, 130])  # Adjusted lower bound of orange color in YUV
        upper_orange = np.array([255, 140, 255])  # Adjusted upper bound of orange color in YUV
        if np.all(colour >= lower_orange) and np.all(colour <= upper_orange):
            return True

    def searchUpDown(self, P0, yuv):
        x0 = P0[0]
        y0 = P0[1]
        x1 = x0
        y1 = y0
        x2 = x0
        y2 = y0
        done = False
        while done == False:
            if self.isTargetColour(yuv[y1 - 3, x1]):
                y1 = y1 - 3
            elif self.isTargetColour(yuv[y1 - 3, x1 - 1]):
                y1 = y1 - 3
                x1 = x1 - 1
            elif self.isTargetColour(yuv[y1 - 3, x1 + 1]):
                y1 = y1 - 3
                x1 = x1 + 1
            else:
                done = True
            if y1 <= 0:
                done = True
                break
        done = False
        while done == False:
            if y2 >= self.height - 3:
                done = True
                break
            if self.isTargetColour(yuv[y2 + 3, x2]):
                y2 = y2 + 3
            elif self.isTargetColour(yuv[y2 + 3, x2 - 1]):
                y2 = y2 + 3
                x2 = x2 - 1
            elif self.isTargetColour(yuv[y2 + 3, x2 + 1]):
                y2 = y2 + 3
                x2 = x2 + 1
            else:
                done = True
            if y2 == self.height - 3:
                done = True
        P1 = (x1, y1)
        P2 = (x2, y2)
        return P1, P2, x1, y1

    def searchLeftRight(self, P0, yuv):
        x0 = P0[0]
        y0 = P0[1]
        xl = x0
        yl = y0
        xr = x0
        yr = y0
        done = False
        while done == False:
            if yl >= self.height - 1 or xl <= 0:
                done = True
                break
            if self.isTargetColour(yuv[yl, xl - 3]):
                xl = xl - 3
            elif self.isTargetColour(yuv[yl - 1, xl - 3]):
                yl = yl - 1
                xl = xl - 3
            elif self.isTargetColour(yuv[yl + 1, xl - 3]):
                yl = yl + 1
                xl = xl - 3
            else:
                done = True
        done = False
        while done == False:
            if yr >= self.height - 1 or xr >= self.width - 3:
                done = True
                break
            if self.isTargetColour(yuv[yr, xr + 2]):
                xr = xr + 3
            elif self.isTargetColour(yuv[yr - 1, xr + 2]):
                yr = yr - 1
                xr = xr + 3
            elif self.isTargetColour(yuv[yr + 1, xr + 3]):
                yr = yr + 1
                xr = xr + 3
            else:
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

    def process(self, inframe, outframe, cinframe = 'None'):
        # Keep this code
        if self.on_jevois:
            bgr = inframe.getCvBGR()
            self.timer.start()
        else:
            bgr = cinframe
        # ADD CODE FROM HERE
        # ---------------------------------------------

        # Convert the image to YUV color space
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        h, w, d, = yuv.shape

        # Define the lower and upper bounds of the orange color in YUV
        max_samples = 100
        sigmaL = 60
        sigma2 = 60

        detectedGates = []
        for i in range(0, max_samples):
            # print(i)
            x0 = randrange(2, w - 3)
            y0 = randrange(80, h - 80)
            P0 = (x0, y0)
            colour0 = yuv[y0, x0]
            if self.isTargetColour(yuv[y0, x0]):
                # highlighted_image[y0, x0] = highlight_color
                P1, P2 = self.searchUpDown(P0, yuv)[:2]
                # print('P1:',P1,'P2:',P2)
                if self.dist(P1, P2) > sigmaL:
                    P3 = self.searchLeftRight(P1, yuv)
                    # print('P3:',P3)
                    P4 = self.searchLeftRight(P2, yuv)
                    # print('P4:',P4)
                    if self.dist(P1, P3) > sigma2 and self.dist(P2,
                                                                P4) > sigma2:  # change and to or when addding refinement filter
                        detectedGate = [P1, P2, P4, P3]
                        # print('P1:', P1, 'P2:', P2)
                        # print('P3:', P3, 'P4:', P4)
                        detectedGates.append(detectedGate)
                        # Draw the detected gate on the image
                        rect = cv2.minAreaRect(np.array(detectedGate))
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.polylines(bgr, [np.array(detectedGate)], isClosed=True, color=(255, 0, 0), thickness=2)

                        cv2.drawContours(bgr, [box], 0, (0, 255, 0), 2)
                        break

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

