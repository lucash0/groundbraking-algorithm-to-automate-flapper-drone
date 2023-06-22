# Fortune favours the bold
from JeVois_code.v02 import Test_v1
from daq_tool import Daq_tool


import cv2
import glob
import numpy as np
import os

if __name__ == "__main__":

    test_data_folder = 'Test_Data\\Flight\\'


    daq = Daq_tool(Test_v1, test_data_folder)
    daq.run_tool()

