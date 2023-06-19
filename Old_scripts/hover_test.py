# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2017-2018 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Version of the AutonomousSequence.py example connecting to 10 Crazyflies.
The Crazyflies go straight up, hover a while and land but the code is fairly
generic and each Crazyflie has its own sequence of setpoints that it files
to.

The layout of the positions:
    x2      x1      x0

y3  10              4

            ^ Y
            |
y2  9       6       3
            |
            +------> X

y1  8       5       2



y0  7               1

"""
import time

import cflib.crtp
import logging
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

logging.basicConfig(level=logging.ERROR)

position_estimate = [0, 0]

data_array = []

# Change uris and sequences according to your setup
URI = uri_helper.uri_from_env(default='radio://0/80/2M/CFE7E7E701')


z0 = 0.4
z = 1.0
x0 = 0
y0 = 0
take_off_alt = z0
landing_alt = 1


#    x   y   z  time
sequence1 = [
    (x0, y0, z0, 3.0),
    (x0, y0, z, 5.0),
    (x0, y0, z0, 3.0),
]


seq_args = {
    URI: [sequence1],
}

# List of URIs, comment the one you do not want to fly
uris = {
    URI,
}


def wait_for_param_download(scf):
    while not scf.cf.param.is_updated:
        time.sleep(1.0)
    print('Parameters downloaded for', scf.cf.link_uri)


def take_off(cf, take_off_alt):
    take_off_time = 3.0
    sleep_time = 0.1
    steps = int(take_off_time / sleep_time)
    vz = take_off_alt / take_off_time

    print(vz)

    for i in range(steps):
        cf.commander.send_velocity_world_setpoint(0, 0, vz, 0)
        time.sleep(sleep_time)


def land(cf, landing_alt):
    landing_time = 3.0
    sleep_time = 0.1
    steps = int(landing_time / sleep_time)
    vz = -landing_alt / landing_time

    print(vz)

    for _ in range(steps):
        cf.commander.send_velocity_world_setpoint(0, 0, vz, 0)
        time.sleep(sleep_time)

    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)


def run_sequence(scf):
    try:
        cf = scf.cf

        take_off(cf, take_off_alt)

        #print('Setting position {}'.format(position))
        end_time = time.time() + 8
        while time.time() < end_time:
            cf.commander.send_zdistance_setpoint(0, 0, 0, -15)
            time.sleep(0.1)
        land(cf, landing_alt)
    except Exception as e:
        print(e)

def log_pos_callback(timestamp, data, logconf):
    data_array.append(data['stateEstimate.z'])
    print(data['stateEstimate.z'])
    global position_estimate


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    cflib.crtp.init_drivers()

    factory = CachedCfFactory(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.z', 'float')

        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        logconf.start()
        run_sequence(scf)
        logconf.stop()
