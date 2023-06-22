# Fortune favours the bold
from JeVois_code.vX import TestModule
from daq_tool import Daq_tool


import cv2
import glob
import numpy as np
import os

if __name__ == "__main__":

    test_data_folder = 'Test_Data\\Flight\\'


    daq = Daq_tool(TestModule, test_data_folder)
    daq.run_tool()

