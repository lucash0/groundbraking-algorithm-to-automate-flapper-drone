# Fortune favours the bold
from JeVois_code.v01 import Test_v1
from daq_tool import Daq_tool


import cv2
import glob
import numpy as np
import os

if __name__ == "__main__":

    daq = Daq_tool(Test_v1)
    daq.run_tool()

