import random
import math


def fx(x):
    if x < 0:
        return 0

    return x


def generate_routefile(num):
    random.seed(100)  # make tests reproducible
    N = num  # number of time steps
    # demand per second from different directions
    with open("roadfile/cross.rou.xml", "w") as routes:
        print(
            """<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="25" guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="1i 2o" />
        <route id="left" edges="2i 1o" />
        <route id="up" edges="3i 4o" />
        <route id="down" edges="4i 3o" />
        <route id="turn_1_3" edges="1i 3o" />
        <route id="turn_1_4" edges="1i 4o" />
        <route id="turn_2_3" edges="2i 3o" />
        <route id="turn_2_4" edges="2i 4o" />
        <route id="turn_3_1" edges="3i 1o" />
        <route id="turn_3_2" edges="3i 2o" />
        <route id="turn_4_1" edges="4i 1o" />
        <route id="turn_4_2" edges="4i 2o" />""",
            file=routes)

        vehNr_r = 0
        vehNr_l = 0
        vehNr_u = 0
        vehNr_d = 0
        vehNr_r_1 = 0
        vehNr_l_1 = 0
        vehNr_u_1 = 0
        vehNr_d_1 = 0
        vehNr_r_2 = 0
        vehNr_l_2 = 0
        vehNr_u_2 = 0
        vehNr_d_2 = 0

        for i in range(N):
            '''pWE = fx(math.sin(i / 100))
            pEW = fx(math.sin(i / 100 + 20))
            pNS = fx(math.sin(i / 100 + 40))
            pSN = fx(math.sin(i / 100 + 60))
            p_turnl = fx(math.sin(i / 100 + 80))
            p_turnr = fx(math.sin(i / 100 + 100))'''
            if i % 100 == 0:
                if int(i / 100) % 2 == 0:
                    pWE = 0.1 * 1
                    pEW = 0.1 * 2
                    pNS = 0.1 * 3
                    pSN = 0.1 * 4
                    p_turnl = 0.1 * 1
                    p_turnr = 0.1 * 2

                if int(i / 100) % 2 == 1:
                    pWE = 0
                    pEW = 0
                    pNS = 0
                    pSN = 0
                    p_turnl = 0
                    p_turnr = 0

            if (random.uniform(0, 1) < pWE):
                print(
                    '    <vehicle id="right_%i" type="typeWE" route="right" departLane="1" arrivalLane="1" depart="%i" />'
                    % (vehNr_r, i * 10),
                    file=routes)
                vehNr_r += 1
                print(
                    '    <vehicle id="right_%i" type="typeWE" route="right" departLane="2" arrivalLane="2" depart="%i" />'
                    % (vehNr_r, i * 10),
                    file=routes)
                vehNr_r += 1

            if (random.uniform(0, 1) < pEW):
                print(
                    '    <vehicle id="_left_%i" type="typeWE" route="left" departLane="1" arrivalLane="1" depart="%i" />'
                    % (vehNr_l, i * 10),
                    file=routes)
                vehNr_l += 1
                print(
                    '    <vehicle id="_left_%i" type="typeWE" route="left" departLane="2" arrivalLane="2" depart="%i" />'
                    % (vehNr_l, i * 10),
                    file=routes)
                vehNr_l += 1

            if (random.uniform(0, 1) < pNS):
                print(
                    '    <vehicle id="_down_%i" type="typeWE" route="down" departLane="1" arrivalLane="1" depart="%i"/>'
                    % (vehNr_d, i * 10),
                    file=routes)
                vehNr_d += 1
                print(
                    '    <vehicle id="_down_%i" type="typeWE" route="down" departLane="2" arrivalLane="2" depart="%i"/>'
                    % (vehNr_d, i * 10),
                    file=routes)
                vehNr_d += 1

            if (random.uniform(0, 1) < pSN):
                print(
                    '    <vehicle id="___up_%i" type="typeWE" route="up" departLane="1" arrivalLane="1" depart="%i"/>' %
                    (vehNr_u, i * 10),
                    file=routes)
                vehNr_u += 1
                print(
                    '    <vehicle id="___up_%i" type="typeWE" route="up" departLane="2" arrivalLane="2" depart="%i"/>' %
                    (vehNr_u, i * 10),
                    file=routes)
                vehNr_u += 1

            if (random.uniform(0, 1) < p_turnl):
                print(
                    '    <vehicle id="turn_1_2_%i" type="typeWE" route="turn_1_4" departLane="3" arrivalLane="3" depart="%i" color="orange"/>'
                    % (vehNr_r_2, i * 10),
                    file=routes)
                vehNr_r_2 += 1

                print(
                    '    <vehicle id="turn_2_2_%i" type="typeWE" route="turn_2_3" departLane="3" arrivalLane="3" depart="%i" color="orange"/>'
                    % (vehNr_l_2, i * 10),
                    file=routes)
                vehNr_l_2 += 1

                print(
                    '    <vehicle id="turn_4_2_%i" type="typeWE" route="turn_4_2" departLane="3" arrivalLane="3" depart="%i" color="orange"/>'
                    % (vehNr_d_2, i * 10),
                    file=routes)
                vehNr_d_2 += 1

                print(
                    '    <vehicle id="turn_3_2_%i" type="typeWE" route="turn_3_1" departLane="3" arrivalLane="3" depart="%i" color="orange"/>'
                    % (vehNr_u_2, i * 10),
                    file=routes)
                vehNr_u_2 += 1

            if (random.uniform(0, 1) < p_turnr):
                print(
                    '    <vehicle id="turn_1_1_%i" type="typeWE" route="turn_1_3" departLane="0" arrivalLane="0" depart="%i" color="blue"/>'
                    % (vehNr_r_1, i * 10),
                    file=routes)
                vehNr_r_1 += 1

                print(
                    '    <vehicle id="turn_2_1_%i" type="typeWE" route="turn_2_4" departLane="0" arrivalLane="0" depart="%i" color="blue"/>'
                    % (vehNr_l_1, i * 10),
                    file=routes)
                vehNr_l_1 += 1

                print(
                    '    <vehicle id="turn_4_1_%i" type="typeWE" route="turn_4_1" departLane="0" arrivalLane="0" depart="%i" color="blue"/>'
                    % (vehNr_d_1, i * 10),
                    file=routes)
                vehNr_d_1 += 1

                print(
                    '    <vehicle id="turn_3_1_%i" type="typeWE" route="turn_3_2" departLane="0" arrivalLane="0" depart="%i" color="blue"/>'
                    % (vehNr_u_1, i * 10),
                    file=routes)
                vehNr_u_1 += 1

        print("</routes>", file=routes)
