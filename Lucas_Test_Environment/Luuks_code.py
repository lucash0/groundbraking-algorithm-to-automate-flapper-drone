import libjevois as jevois
import cv2
import numpy as np
import random
import string
import os

class Test_v1:
    def __init__(self):
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        self.frame = 0
        self.output_dir = None

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

    def process(self, inframe, outframe):
        bgr = inframe.getCvBGR()
        self.timer.start()

        # Convert the image to YUV color space
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        # Define the lower and upper bounds of the orange color in YUV
        lower_orange = np.array([0, 0, 160])  # Lower bound of orange color in YUV
        upper_orange = np.array([255, 120, 255])  # Upper bound of orange color in YUV

        # Threshold the image to get a binary mask of the orange color
        mask = cv2.inRange(yuv, lower_orange, upper_orange)

        # Find contours of the orange in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a copy of the original image for drawing
            output_image = bgr.copy()

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
            outframe.sendCv(bgr)

    def processNoUSB(self, inframe, outframe):
        bgr = inframe.getCvBGR()
        self.timer.start()

        # Convert the image to YUV color space
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        # Define the lower and upper bounds of the orange color in YUV
        lower_orange = np.array([0, 0, 160])  # Lower bound of orange color in YUV
        upper_orange = np.array([255, 120, 255])  # Upper bound of orange color in YUV

        # Threshold the image to get a binary mask of the orange color
        mask = cv2.inRange(yuv, lower_orange, upper_orange)

        # Find contours of the orange in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a copy of the original image for drawing
            output_image = bgr.copy()

            # Draw bounding rectangle around the largest contour and add centroid dot
            x, y, w, h = cv2.boundingRect(largest_contour)
            # cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw a dot at the centroid
                # cv2.circle(output_image, (cX, cY), 3, (0, 0, 255), -1)

        if self.output_dir is None:
            self.create_output_dir()

        # Save the processed image
        cv2.imwrite(f'{self.output_dir}/img{self.frame}.png', output_image)
        self.frame += 1

        fps = self.timer.stop()
        height, width, _ = output_image.shape
            # cv2.putText(output_image, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
