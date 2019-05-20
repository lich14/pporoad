'''the first program change old state to sparse state
tftime=[0,60]
self.gamma = 0.95
learningrate=0.001
add entropie
god bless me
'''

from __future__ import absolute_import, print_function

import csv
import math
import optparse
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

parser = argparse.ArgumentParser(description='Train a PPO agent for traffic light')
parser.add_argument('--bound', type=float, default=0)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transition = np.dtype([('s', np.float64, (4, 28, 28)), ('a', np.float64, (4,)), ('r', np.float64),
                       ('s_', np.float64, (4, 28, 28)), ('a_logp', np.float64)])


class store():
    buffer_capacity, batch_size = 256, 32

    def __init__(self):
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

    def store(self, add):
        self.buffer[self.counter] = add
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False


class PPOnet(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(PPOnet, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(4, 8, kernel_size=2, stride=1),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=2, stride=1),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2))
        self.v = nn.Sequential(nn.Linear(576, 2000), nn.ReLU(), nn.Linear(2000, 2000), nn.ReLU(), nn.Linear(2000, 1))
        self.fc = nn.Sequential(nn.Linear(576, 2000), nn.ReLU(), nn.Linear(2000, 2000), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(2000, 4), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(2000, 4), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 576)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class sumomodel():

    def __init__(self):
        self.net = PPOnet().double().to(device)
        self.optimizer = optim.Adam(self.net.parameters())
        self.clip_param = 0.1
        self.PPOepoch = 8
        self.memory = store()
        self.gamma = 0.95
        self.path_t7 = './checkpoint/p_adam' + str(args.bound) + '.t7'
        self.path_wt = './csvfiles/averange_adam' + str(args.bound) + '.csv'
        self.path_lsa = './csvfiles/loss_a_adam' + str(args.bound) + '.csv'
        self.path_lsv = './csvfiles/loss_v_adam' + str(args.bound) + '.csv'
        self.path_ac = './csvfiles/ac_adam' + str(args.bound) + '.csv'

        self.step = 0
        self.loop = 0
        self.dict1 = {}
        self.periodtime = []
        self.currentstate = 0
        self.time_click = 0
        self.out_record = None
        self.max_grad_norm = 0.5
        self.wtime = []
        self.tflightime = np.array([30, 30, 30, 30])

        if os.path.exists("./csvfiles/") is False:
            os.makedirs("./csvfiles/")
        if os.path.exists("./checkpoint/") is False:
            os.makedirs("./checkpoint/")
        if os.path.isfile(self.path_t7):
            self.net.load_state_dict(torch.load(self.path_t7))

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def writeac(self, action):
        if os.path.isfile(self.path_ac):
            with open(self.path_ac, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(action)
        else:
            with open(self.path_ac, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['t1', 't2', 't3', 't4'])
                csv_write.writerow(action)

    def getstate(self, list_car):
        vehicle_record = np.zeros([28, 28])

        for k in list_car:
            if (traci.vehicle.getLaneID(k)[1] is 'i'):
                num1 = int(traci.vehicle.getLaneID(k)[0])
                num2 = int(traci.vehicle.getLaneID(k)[3])
                bias = num2 - 3.5
                if num1 % 2 == 1:
                    bias = -1 * bias + 13.5
                else:
                    bias = bias + 13.5

                if (num2 != 0):
                    pos = min(math.floor(int(traci.vehicle.getLanePosition(k)) / 50), 9)
                    bias = int(bias)
                    if num1 == 1:
                        vehicle_record[pos, bias] += 1
                    if num1 == 2:
                        vehicle_record[27 - pos, bias] += 1
                    if num1 == 3:
                        vehicle_record[bias, 27 - pos] += 1
                    if num1 == 4:
                        vehicle_record[bias, pos] += 1

        return vehicle_record

    def trainmodel(self):
        s = torch.tensor(self.memory.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.memory.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.memory.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.memory.buffer['s_'], dtype=torch.double).to(device)
        r = (r - r.mean()) / (r.std() + 1e-5)
        old_a_logp = torch.tensor(self.memory.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]

        for _ in range(self.PPOepoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.memory.buffer_capacity)), self.memory.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1)
                a_logp = a_logp.reshape(-1, 1)
                ratio = torch.exp(a_logp - old_a_logp[index])
                with torch.no_grad():
                    entrop = dist.entropy()

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                self.storeloss(action_loss, value_loss)
                action_loss = torch.clamp(action_loss, 0, 10)
                value_loss = torch.clamp(value_loss, 0, 10)
                loss = action_loss + 2. * value_loss - args.bound * entrop.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        torch.save(self.net.state_dict(), self.path_t7)

    def storeloss(self, action_loss, value_loss):
        if os.path.isfile(self.path_lsa):
            with open(self.path_lsa, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [float(action_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)
        else:
            with open(self.path_lsa, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['loss'])
                data_row = [float(action_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)

        if os.path.isfile(self.path_lsv):
            with open(self.path_lsv, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [float(value_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)
        else:
            with open(self.path_lsv, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['loss'])
                data_row = [float(value_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)

    def writetime(self, a):
        path = self.path_wt
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([a])
        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['averangetime'])
                csv_write.writerow([a])

    def run(self):
        sumoBinary = checkBinary('sumo')
        traci.start([sumoBinary, "-c", "roadfile/cross.sumocfg"])
        listpic = []
        self.loop = 0
        self.dict1 = {}
        self.periodtime = []
        self.currentstate = 0
        self.time_click = 0
        self.out_record = None
        self.max_grad_norm = 0.5
        self.wtime = []
        self.tflightime = np.array([30, 30, 30, 30])

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if (self.currentstate is not traci.trafficlight.getPhase("0")):
                self.time_click = 0
            self.time_click = self.time_click + 1

            phase_index = int(traci.trafficlight.getPhase("0") / 2)
            list_car = traci.vehicle.getIDList()
            for k in list_car:
                traci.vehicle.setLaneChangeMode(k, 0b001000000000)

            vehiclein_l = traci.simulation.getDepartedIDList()
            if (vehiclein_l):
                for i in vehiclein_l:
                    self.dict1[i] = self.step

            vehicleout_l = traci.simulation.getArrivedIDList()
            if (vehicleout_l):
                for i in vehicleout_l:
                    self.periodtime.append(self.step - self.dict1[i])
                    self.wtime.append(self.step - self.dict1[i])
                    self.dict1.pop(i)

            if ((self.step - int(self.step / 2000) * 2000) % 5 == 0 and int(
                (self.step - int(self.step / 2000) * 2000) / 5) < 4):
                listpic.append(self.getstate(list_car))

            if self.step % 1000 == 999:
                if int(self.step / 1000) % 2 == 0:
                    if (self.wtime):
                        self.writetime(np.array(self.wtime).mean())

                self.wtime = []

            if (self.step % 2000 == 15):
                print(self.loop)
                self.loop = self.loop + 1
                if (len(listpic) != 4):
                    break
                pict = np.array(listpic)
                input_d = torch.tensor(pict, dtype=torch.double).to(device)
                input_d = input_d.reshape(1, 4, 28, 28)
                listpic = []

                with torch.no_grad():
                    alpha, beta = self.net(input_d)[0]

                dist = Beta(alpha, beta)
                action = dist.sample()
                a_logp = dist.log_prob(action.view(-1, 4)).sum(dim=1)

                action = action.squeeze().cpu().numpy()
                a_logp = a_logp.item()
                self.tflightime = np.array(action * 60)
                self.writeac(self.tflightime.tolist())

                reward = 0
                if (self.periodtime):
                    reward = -0.9 * np.array(self.periodtime).mean() - 0.1 * np.array(self.periodtime).max()

                self.periodtime = []

                ifupdata = None
                if self.out_record is not None:
                    ifupdata = self.memory.store((self.out_record[0], self.out_record[1], reward, pict,
                                                  self.out_record[2]))
                self.out_record = [pict, action, a_logp]

                if ifupdata is True:
                    print('train')
                    self.trainmodel()

            self.currentstate = traci.trafficlight.getPhase("0")
            if (self.time_click >= self.tflightime[phase_index]):
                traci.trafficlight.setPhase("0", (self.currentstate + 1) % 8)

            self.step += 1

        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    p = sumomodel()
    while (1):
        p.run()
