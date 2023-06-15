import libjevois as jevois
import cv2
import numpy as np
import os

class TestModule:
    def __init__(self):
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        self.frame = 0
        self.output_dir = None

    def create_output_dir(self):
        folder_number = 1
        base_dir = 'data/test'
        current_dir = base_dir + str(folder_number)
        
        while os.path.isdir(current_dir):
            folder_number += 1
            current_dir = base_dir + str(folder_number)
            
        os.mkdir(current_dir)
        self.output_dir = current_dir

    def process(self, inframe, outframe):
        img = inframe.getCvBGR()

        if self.output_dir is None:
            self.create_output_dir()

        # Save the processed image
        cv2.imwrite(f'{self.output_dir}/img{self.frame}.png', img)
        self.frame += 1

        outframe.sendCv(img)
            
            
    def processNoUSB(self, inframe):
        img = inframe.getCvBGR()

        if self.output_dir is None:
            self.create_output_dir()

        # Save the processed image
        cv2.imwrite(f'{self.output_dir}/img{self.frame}.png', img)
        self.frame += 1