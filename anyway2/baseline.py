#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2018 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html
# SPDX-License-Identifier: EPL-2.0

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$

from __future__ import absolute_import
from __future__ import print_function

import roadcreate
import os
import sys
import optparse
import numpy as np
import csv
import math
import threading

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

if os.path.exists("./csvfiles/") is False:
    os.makedirs("./csvfiles/")
if os.path.exists("./checkpoint/") is False:
    os.makedirs("./checkpoint/")


def getstate(list_car):
    vehicle_record = np.zeros([28, 28])

    for k in list_car:
        if (traci.vehicle.getLaneID(k)[1] is 'i'):
            num1 = int(traci.vehicle.getLaneID(k)[0])
            num2 = int(traci.vehicle.getLaneID(k)[3])
            bias = num2 - 3.5
            if num1 == 1 or num1 == 4:
                bias = -1 * bias + 13.5
            if num1 == 2 or num1 == 3:
                bias = bias + 13.5

            if (num2 != 0):
                pos = min(math.floor(int(traci.vehicle.getLanePosition(k)) / 50), 9)
                bias = int(bias)
                if num1 == 1:
                    vehicle_record[bias, pos] += 1
                if num1 == 2:
                    vehicle_record[bias, 27 - pos] += 1
                if num1 == 3:
                    vehicle_record[pos, bias] += 1
                if num1 == 4:
                    vehicle_record[27 - pos, bias] += 1

    return vehicle_record


def run():
    """execute the TraCI control loop"""
    step = 0

    currentstate = 0
    time_click = 0
    loop = 0
    dict1 = {}
    periodtime = []
    tflightime = np.array([30, 30, 30, 30])
    '''for transmission in tf state'''
    '''store_meanspeed = []
    list_edge = ['1i', '2i', '3i', '4i']'''
    '''DQN action=[WE add 5s, WE minus 5s, WE left add 5s, WE left minus 5s, 
                   NS add 5s, NS minus 5s, NS left add 5s, NS left minus 5s] '''

    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()
        vehiclein_l = traci.simulation.getDepartedIDList()

        if (vehiclein_l):
            for i in vehiclein_l:
                dict1[i] = step

        vehicleout_l = traci.simulation.getArrivedIDList()
        if (vehicleout_l):
            for i in vehicleout_l:
                periodtime.append(step - dict1[i])

        if (currentstate is not traci.trafficlight.getPhase("0")):
            time_click = 0
        time_click = time_click + 1

        currentphase = traci.trafficlight.getPhase("0")
        phase_index = int(currentphase / 2)
        list_car = traci.vehicle.getIDList()
        for k in list_car:
            traci.vehicle.setLaneChangeMode(k, 0b001000000000)

        path = "./csvfiles/averange_baseline_30_100.csv"
        if step % 1000 == 999:
            if int(step / 1000) % 2 == 0:
                if os.path.isfile(path):
                    with open(path, 'a+') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow([np.array(periodtime).mean()])
                else:
                    with open(path, 'w') as f:
                        csv_write = csv.writer(f)
                        csv_write.writerow(['averangetime'])
                        csv_write.writerow([np.array(periodtime).mean()])

            periodtime = []

        if (currentstate is 7 and traci.trafficlight.getPhase("0") is 0):
            print(loop)
            loop = loop + 1

        currentstate = traci.trafficlight.getPhase("0")
        if (time_click >= tflightime[phase_index]):
            traci.trafficlight.setPhase("0", (currentphase + 1) % 8)

        step += 1
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "--nogui",
        --device.rerouting.threads2,
        action="store_true",
        default=False,
        help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def ttt(a):
    options = get_options()
    sumoBinary = checkBinary('sumo')
    str1 = "roadfile" + a + "/cross.sumocfg"
    str2 = "roadfile" + a + "/tripinfo.xml"
    print(str1, str2)
    traci.start([sumoBinary, "-c", str1, "--tripinfo-output", str2])
    run()


# this is the main entry point of this script
if __name__ == "__main__":
    roadcreate.generate_routefile(100000)
    '''threads = []
    t1 = threading.Thread(target=ttt, args=['1'])
    threads.append(t1)
    t2 = threading.Thread(target=ttt, args=['2'])
    threads.append(t2)
    t1.start()
    t2.start()
    for i in threads:
        i.join()'''

    ttt('1')
