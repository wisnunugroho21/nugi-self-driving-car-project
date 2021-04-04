import os
import sys
import optparse
import random
from gym import spaces

import numpy as np

sys.path.append('/usr/share/sumo/tools')
from sumolib import checkBinary
import traci

class SumoEnv:  
    def __init__(self):        
        self.time = 0
        self.run = False
        
        self.waktu_merah_kanan = 1
        self.waktu_merah_bawah = 1
        self.waktu_merah_kiri = 1
        
        self.__generate_routefile("environment/sumo/test1.rou.xml") # first, generate the route file for this simulation

        self.observation_space  = spaces.Box(-100, 100, (12, ))
        self.action_space       = spaces.Discrete(3)
        
    def __generate_routefile(self, route_files):
        random.seed(10)  # make tests reproducible
        N = 300  # number of time steps
        # demand per second from different directions
        pLR = 1. / 8
        pRL = 1. / 8
        pLU = 1. / 20
        pRU = 1. / 20
        pUL = 1. / 24
        pUR = 1. / 24
        with open(route_files, "w") as routes:
            print("""<routes>
            <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
    guiShape="passenger"/>
            <vType id="bus" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

            <route id="left_right" edges="gneE4 gneE1"/>
            <route id="right_left" edges="gneE5 gneE0"/>
            <route id="left_down" edges="gneE4 gneE2"/>
            <route id="right_down" edges="gneE5 gneE2"/>
            <route id="down_left" edges="gneE6 gneE0"/>
            <route id="down_right" edges="gneE6 gneE1"/>""", file=routes)
            vehNr = 0
            
            for i in range(N):
                if random.uniform(0, 1) < pLR:
                    print('    <vehicle id="left_right_%i" type="car" route="left_right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pRL:
                    print('    <vehicle id="right_left_%i" type="car" route="right_left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pLU:
                    print('    <vehicle id="left_down_%i" type="car" route="left_down" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pRU:
                    print('    <vehicle id="right_down_%i" type="car" route="right_down" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pUL:
                    print('    <vehicle id="down_left_%i" type="car" route="down_left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pUR:
                    print('    <vehicle id="down_right_%i" type="car" route="down_right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    
            print("</routes>", file=routes)
    
    def reset(self):
        sumoBinary = checkBinary('sumo-gui')
        
        if self.run:
            traci.close()  
            
        traci.start([sumoBinary, "-c", "environment/sumo/test1.sumocfg", 
        "--tripinfo-output", "environment/sumo/test1.xml",
        "--no-step-log",
        "--no-warnings",
        "--duration-log.disable"])
        self.run = True
        
        self.time = 0
        self.waktu_merah_bawah = 1
        self.waktu_merah_kanan = 1
        self.waktu_merah_kiri = 1

        return np.zeros(12)
    
    def step(self, action):
        reward = 0
        traci.trafficlight.setPhase("tl_1", action)
        traci.simulationStep()

        banyak_kendaraan_tabrakan = traci.simulation.getCollidingVehiclesNumber()

        banyak_kendaraan_bawah1 = traci.lanearea.getLastStepVehicleNumber('detektor_bawah_1')
        banyak_kendaraan_bawah2 = traci.lanearea.getLastStepVehicleNumber('detektor_bawah_2')
        
        banyak_kendaraan_kanan1 = traci.lanearea.getLastStepVehicleNumber('detektor_kanan_1')
        banyak_kendaraan_kanan2 = traci.lanearea.getLastStepVehicleNumber('detektor_kanan_2')

        banyak_kendaraan_kiri1 = traci.lanearea.getLastStepVehicleNumber('detektor_kiri_1')
        banyak_kendaraan_kiri2 = traci.lanearea.getLastStepVehicleNumber('detektor_kiri_2')
        
        panjang_antrian_bawah1 = traci.lanearea.getLastStepHaltingNumber('detektor_bawah_1') 
        panjang_antrian_bawah2 = traci.lanearea.getLastStepHaltingNumber('detektor_bawah_2')

        panjang_antrian_kanan1 = traci.lanearea.getLastStepHaltingNumber('detektor_kanan_1')
        panjang_antrian_kanan2 = traci.lanearea.getLastStepHaltingNumber('detektor_kanan_2')

        panjang_antrian_kiri1 = traci.lanearea.getLastStepHaltingNumber('detektor_kiri_1')
        panjang_antrian_kiri2 = traci.lanearea.getLastStepHaltingNumber('detektor_kiri_2')

        kecepatan_kendaraan_bawah = (traci.inductionloop.getLastStepMeanSpeed('detektor_titik_bawah_1') + traci.inductionloop.getLastStepMeanSpeed('detektor_titik_bawah_2')) / 2
        kecepatan_kendaraan_kanan = (traci.inductionloop.getLastStepMeanSpeed('detektor_titik_kanan_1') + traci.inductionloop.getLastStepMeanSpeed('detektor_titik_kanan_2')) / 2
        kecepatan_kendaraan_kiri = (traci.inductionloop.getLastStepMeanSpeed('detektor_titik_kiri_1') + traci.inductionloop.getLastStepMeanSpeed('detektor_titik_kiri_2')) / 2
        
        banyak_kendaraan_lewat_bawah = traci.inductionloop.getLastStepVehicleNumber('detektor_titik_bawah_1') + traci.inductionloop.getLastStepVehicleNumber('detektor_titik_bawah_2')
        banyak_kendaraan_lewat_kanan = traci.inductionloop.getLastStepVehicleNumber('detektor_titik_kanan_1') + traci.inductionloop.getLastStepVehicleNumber('detektor_titik_kanan_2')
        banyak_kendaraan_lewat_kiri = traci.inductionloop.getLastStepVehicleNumber('detektor_titik_kiri_1') + traci.inductionloop.getLastStepVehicleNumber('detektor_titik_kiri_2')

        if action == 2:
            self.waktu_merah_bawah = 1
            self.waktu_merah_kanan += 1
            self.waktu_merah_kiri += 1

        elif action == 1:
            self.waktu_merah_kanan = 1
            self.waktu_merah_bawah += 1
            self.waktu_merah_kiri += 1

        elif action == 0:
            self.waktu_merah_kiri = 1
            self.waktu_merah_bawah += 1
            self.waktu_merah_kanan += 1
            
        reward -= (self.waktu_merah_bawah * 0.2) + ((panjang_antrian_bawah1 + panjang_antrian_bawah2) * 0.5)
        reward -= (self.waktu_merah_kanan * 0.2) + ((panjang_antrian_kanan1 + panjang_antrian_kanan2) * 0.5)
        reward -= (self.waktu_merah_kiri * 0.2) + ((panjang_antrian_kiri1 + panjang_antrian_kiri2) * 0.5)        
        reward -= (banyak_kendaraan_tabrakan * 5.0)
        reward += (banyak_kendaraan_lewat_bawah * 2.0) + (banyak_kendaraan_lewat_kanan * 2.0) + (banyak_kendaraan_lewat_kiri * 2.0)
        
        self.time += 1
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        if not done:
            done = self.time > 100000

        obs = np.array([panjang_antrian_bawah1, panjang_antrian_bawah2, banyak_kendaraan_bawah1, banyak_kendaraan_bawah2,
                         panjang_antrian_kanan1, panjang_antrian_kanan2, banyak_kendaraan_kanan1, banyak_kendaraan_kanan2,
                         panjang_antrian_kiri1, panjang_antrian_kiri2, banyak_kendaraan_kiri1, banyak_kendaraan_kiri2])

        info = 0
            
        return obs, reward, done, info