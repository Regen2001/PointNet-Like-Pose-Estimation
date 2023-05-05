import numpy as np
from copy import deepcopy
import random
import os
import shutil
from tqdm import tqdm
import threading
from Cube import Cube
from Cylinder import Cylinder
from H_structure import H_structure
from Double_cube import Double_cube
from Double_cylinder import Double_cylinder
from Cube_cylinder import Cube_cylinder

number_item = 5

class THREAD(threading.Thread):
    def __init__(self, func, args) :
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(self.args)

def main():
    if os.path.exists('data') is True:
        shutil.rmtree('data')
        os.makedirs('data')
    else:
        os.makedirs('data')

    os.makedirs('data/cube')
    os.makedirs('data/cuboid')
    os.makedirs('data/cylinder')
    os.makedirs('data/h_structure')
    os.makedirs('data/double_cube')
    os.makedirs('data/double_cylinder')
    os.makedirs('data/cube_cylinder')

    threads = list()
    threads.append(THREAD(generate_cube, 1))
    threads.append(THREAD(generate_cuboid, 1))
    threads.append(THREAD(generate_cylinder, 1))
    threads.append(THREAD(generate_h_structure, 1))
    threads.append(THREAD(generate_double_cube, 1))
    threads.append(THREAD(generate_double_cylinder, 1))
    threads.append(THREAD(generate_cube_cylinder, 1))

    print('Begin to generate the data')

    for i in threads:
        i.start()

    for i in threads:
        i.join()
    
    print('Finish to generate the data')

    return 0

def generate_cube(start_number=1):
    for i in tqdm(range(start_number,start_number+number_item)):
        size = random.randint(5,10) / 100
        px = random.randint(-50,50) / 100
        py = random.randint(-50,50) / 100
        pz = random.randint(-100,-60) / 100
        phi = random.randint(-45,45)
        theta = random.randint(-45,45)
        psi = random.randint(-45,45)
        points = Cube(size, size, size)
        points.rotation([phi, theta, psi])
        points.translation([px, py, pz])
        points.savePoints('cube', i)

def generate_cuboid(start_number=1):
    for i in tqdm(range(start_number,start_number+number_item)):
        length = random.randint(3,6) / 100
        width = random.randint(7,10) / 100
        height= random.randint(4,8) / 100
        px = random.randint(-50,50) / 100
        py = random.randint(-50,50) / 100
        pz = random.randint(-100,-60) / 100
        phi = random.randint(-90,90)
        theta = random.randint(-45,45)
        psi = random.randint(-45,45)
        points = Cube(length, width, height)
        points.rotation([phi, theta, psi])
        points.translation([px, py, pz])
        points.savePoints('cuboid', i)

def generate_cylinder(start_number=1):
    for i in tqdm(range(start_number,start_number+number_item)):
        radius = random.randint(15,40) / 1000
        height = random.randint(4,8) / 100
        px = random.randint(-50,50) / 100
        py = random.randint(-50,50) / 100
        pz = random.randint(-100,-60) / 100
        phi = random.randint(-90,90)
        theta = random.randint(-45,45)
        psi = random.randint(-45,45)
        points = Cylinder(radius, height)
        points.rotation([phi, theta, psi])
        points.translation([px, py, pz])
        points.savePoints(i)

def generate_h_structure(start_number=1):
    for i in tqdm(range(start_number,start_number+number_item)):
        h = random.randint(8,10) / 100
        b = random.randint(8,10) / 100
        t_1 = random.randint(1,3) / 100
        t_2= random.randint(1,3) / 100
        height= random.randint(5,10) / 100
        px = random.randint(-50,50) / 100
        py = random.randint(-50,50) / 100
        pz = random.randint(-100,-60) / 100
        phi = random.randint(-90,90)
        theta = random.randint(-45,45)
        psi = random.randint(-45,45)
        points = H_structure(h, b, t_1, t_2, height)
        points.rotation([phi, theta, psi])
        points.translation([px, py, pz])
        points.savePoints(i)

def generate_double_cube(start_number=1):
    for i in tqdm(range(start_number,start_number+number_item)):
        size_1 = [[0.08, 0.03, 0.05], [0.04, 0.06, 0.08], [0.02, 0.07,0.03]]
        size_2 = [[0.08, 0.03, 0.08], [0.07, 0.09, 0.09], [0.10, 0.04,0.09]]
        SIZE_1 = random.randint(0,2)
        SIZE_2 = random.randint(0,2)
        px = random.randint(-50,50) / 100
        py = random.randint(-50,50) / 100
        pz = random.randint(-100,-60) / 100
        phi = random.randint(-90,90)
        theta = random.randint(-45,45)
        psi = random.randint(-45,45)
        points = Double_cube(size_1[SIZE_1], size_2[SIZE_2],  excursion=True)
        points.rotation([phi, theta, psi])
        points.translation([px, py, pz])
        points.savePoints(i)

def generate_double_cylinder(start_number=1):
    for i in tqdm(range(start_number,start_number+number_item)):
        size_1 = [[0.02, 0.07], [0.03, 0.06], [0.015, 0.09]]
        size_2 = [[0.04, 0.08], [0.05, 0.10], [0.035, 0.09]]
        SIZE_1 = random.randint(0,2)
        SIZE_2 = random.randint(0,2)
        px = random.randint(-50,50) / 100
        py = random.randint(-50,50) / 100
        pz = random.randint(-100,-60) / 100
        phi = random.randint(-90,90)
        theta = random.randint(-45,45)
        psi = random.randint(-45,45)
        points = Double_cylinder(size_1[SIZE_1], size_2[SIZE_2],  excursion=True)
        points.rotation([phi, theta, psi])
        points.translation([px, py, pz])
        points.savePoints(i)

def generate_cube_cylinder(start_number=1):
    for i in tqdm(range(start_number,start_number+number_item)):
        size_1 = [[0.02, 0.07], [0.03, 0.06], [0.015, 0.09]]
        size_2 = [[0.08, 0.05, 0.08], [0.07, 0.09, 0.09], [0.10, 0.04,0.09]]
        SIZE_1 = random.randint(0,2)
        SIZE_2 = random.randint(0,2)
        px = random.randint(-50,50) / 100
        py = random.randint(-50,50) / 100
        pz = random.randint(-100,-60) / 100
        phi = random.randint(-90,90)
        theta = random.randint(-45,45)
        psi = random.randint(-45,45)
        points = Cube_cylinder(size_1[SIZE_1], size_2[SIZE_2],  excursion=True)
        points.rotation([phi, theta, psi])
        points.translation([px, py, pz])
        points.savePoints(i)

if __name__ == '__main__':
    main()