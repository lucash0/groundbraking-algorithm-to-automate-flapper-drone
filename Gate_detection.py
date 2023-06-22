import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from random import randrange
import math


class Test_v1:
    def __init__(self):
        self.output_dir = 'images_processed'
        self.frame = 0

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def isTargetColour(self, colour):
        lower_orange = np.array([0, 0, 130])  # Adjusted lower bound of orange color in YUV
        upper_orange = np.array([255, 140, 255])  # Adjusted upper bound of orange color in YUV
        if np.all(colour >= lower_orange) and np.all(colour <= upper_orange):
            return True

    def dist(self, P1, P2):
        x1, y1 = P1
        x2, y2 = P2
        dx = abs(x1-x2)
        dy = abs(y1-y2)
        distance = math.sqrt(dx**2 + dy**2)
        return distance

    def searchUpDown(self, P0, yuv):
        x0 = P0[0]
        y0 = P0[1]
        x1 = x0
        y1 = y0
        x2 = x0
        y2 = y0
        done = False
        while done == False:
            if self.isTargetColour(yuv[y1 - 1, x1]):
                y1 = y1 - 1
            elif self.isTargetColour(yuv[y1 - 1, x1 - 1]):
                y1 = y1 - 1
                x1 = x1 - 1
            elif self.isTargetColour(yuv[y1 - 1, x1 + 1]):
                y1 = y1 - 1
                x1 = x1 + 1
            else:
                done = True
            if y1 == 0:
                done = True
        done = False
        while done == False:
            if self.isTargetColour(yuv[y2 + 1, x2]):
                y2 = y2 + 1
            elif self.isTargetColour(yuv[y2 + 1, x2 - 1]):
                y2 = y2 + 1
                x2 = x2 - 1
            elif self.isTargetColour(yuv[y2 + 1, x2 + 1]):
                y2 = y2 + 1
                x2 = x2 + 1
            else:
                done = True
            if y2 == 239:
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
            if self.isTargetColour(yuv[yl, xl - 1]):
                xl = xl - 1
            elif self.isTargetColour(yuv[yl - 1, xl - 1]):
                yl = yl - 1
                xl = xl - 1
            elif self.isTargetColour(yuv[yl + 1, xl - 1]):
                yl = yl + 1
                xl = xl - 1
            else:
                done = True
            if yl == 239:
                done = True
        done = False
        while done == False:
            if yr == 239:
                done = True
            if self.isTargetColour(yuv[yr, xr + 1]):
                xr = xr + 1
            elif self.isTargetColour(yuv[yr - 1, xr + 1]):
                yr = yr - 1
                xr = xr + 1
            elif self.isTargetColour(yuv[yr + 1, xr + 1]):
                yr = yr + 1
                xr = xr + 1
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


    def snake_detection(self, image_path):
        bgr = cv2.imread(image_path)

        # Convert the image to YUV color space
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        h, w, d, = yuv.shape

        # Define the lower and upper bounds of the orange color in YUV
        max_samples = 8000
        sigmaL = 60

        detectedGates = []
        for i in range(0, max_samples):
            # print(i)
            x0 = randrange(2, w - 3)
            y0 = randrange(4, h - 3)
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
                    if self.dist(P1, P3) > sigmaL and self.dist(P2,P4) > sigmaL:  # change and to or when addding refinement filter
                        detectedGate = [P1, P2, P4, P3]
                        #print('P1:', P1, 'P2:', P2)
                        #print('P3:', P3, 'P4:', P4)
                        detectedGates.append(detectedGate)
                        # Draw the detected gate on the image
                        cv2.polylines(bgr, [np.array(detectedGate)], isClosed=True, color=(0, 255, 0), thickness=2)
                        break

        # Save the processed image
        self.create_output_dir()
        output_path = os.path.join(self.output_dir, f'img{self.frame}.png')
        cv2.imwrite(output_path, bgr)
        print(output_path)
        self.frame += 1
        if self.frame == 545:
            return


    def process_images_in_folder(self, folder_path):
        image_files = os.listdir(folder_path)
        sorted_image_files = sorted(image_files, key=lambda x: int(x[3:-4]))
        for image_file in sorted_image_files:
            image_path = os.path.join(folder_path, image_file)
            self.snake_detection(image_path)


test = Test_v1()
test.process_images_in_folder('Test_Data\Flight')