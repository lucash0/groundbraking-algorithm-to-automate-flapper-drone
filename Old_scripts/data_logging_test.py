import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

from matplotlib import pyplot as plt
import numpy as np


URI = uri_helper.uri_from_env(default='radio://0/80/2M/CFE7E7E701')

DEFAULT_HEIGHT = 0.5

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

position_estimate = [0, 0]

data_array = []

# def move_linear_simple(scf):
#     with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
#         time.sleep(1)
#         mc.forward(0.5)
#         time.sleep(1)
#         mc.turn_left(180)
#         time.sleep(1)
#         mc.forward(0.5)
#         time.sleep(1)


def take_off_simple(scf):
    with MotionCommander(scf, default_height=-16) as mc:
        time.sleep(3)
        mc.stop()

def log_pos_callback(timestamp, data, logconf):
    data_array.append(data['stateEstimate.z'])
    print(data['stateEstimate.z'])
    global position_estimate
    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']


# def param_deck_flow(name, value_str):
#     value = int(value_str)
#     print(value)
#     if value:
#         deck_attached_event.set()
#         print('Deck is attached!')
#     else:
#         print('Deck is NOT attached!')

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        # scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
        #                                  cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')

        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        # if not deck_attached_event.wait(timeout=0.1):
        #     print('No flow deck detected!')
            # sys.exit(1)

        logconf.start()

        take_off_simple(scf)
        logconf.stop()

    print(data_array)
    x_array = np.arange(0, (len(data_array) + 1))
    plt.title("Altitude graph")
    plt.plot(x_array, data_array)
    plt.show()

    



