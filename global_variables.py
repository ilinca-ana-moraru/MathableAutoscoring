import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

GLOBAL_IN = "test_data"
GLOBAL_OUT = os.path.join("test_data","fisiere_solutie","363_Moraru_Ilinca")
nr_of_games = 4

def show_image(image):
    plt.figure(dpi = 20)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()

lines_horizontal=[]
for i in range(0,1500,100):
    l=[]
    l.append((0,i))
    l.append((1398,i))
    lines_horizontal.append(l)

lines_vertical=[]
for i in range(0,1500,100):
    l=[]
    l.append((i,0))
    l.append((i,1398))
    lines_vertical.append(l)

class box:
    def __init__(self, top_left, top_right, bottom_left, bottom_right, area):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right      
        self.area = area  

class cut_digit:
    def __init__(self, top_x, bottom_x, left_y, right_y, area, image):
        self.top_x = top_x
        self.bottom_x = bottom_x
        self.left_y = left_y
        self.right_y = right_y
        self.area = area        
        self.image = image

board_width = 1400
board_height = 1400

board_coordonates = box((268,263),(1734,269),(262,1739),(1736,1749),0)

class_to_cluster_idx = [4,3,6,5,2,7,1,0,8,9]  

biggest_width = 36
biggest_height = 55

BOARD_PIECES = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,1,2,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,3,4,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
]

BONUSES = [
    [3,1,1,1,1,1,3,3,1,1,1,1,1,3],
    [1,2,1,1,1,1,1,1,1,1,1,1,2,1],
    [1,1,2,1,1,1,1,1,1,1,1,2,1,1],
    [1,1,1,2,1,1,1,1,1,1,2,1,1,1],
    [1,1,1,1,2,1,1,1,1,2,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [3,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [3,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,2,1,1,1,1,2,1,1,1,1],
    [1,1,1,2,1,1,1,1,1,1,2,1,1,1],
    [1,1,2,1,1,1,1,1,1,1,1,2,1,1],
    [1,2,1,1,1,1,1,1,1,1,1,1,2,1],
    [3,1,1,1,1,1,3,3,1,1,1,1,1,3]
]


A = 1
S = 2
M = 3
D = 4

CONSTRAINTS = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,D,0,0,0,0,D,0,0,0,0],
    [0,0,0,0,0,S,0,0,S,0,0,0,0,0],
    [0,0,0,0,0,0,A,M,0,0,0,0,0,0],
    [0,D,0,0,0,0,M,A,0,0,0,0,D,0],
    [0,0,S,0,0,0,0,0,0,0,0,S,0,0],
    [0,0,0,M,A,0,0,0,0,M,A,0,0,0],
    [0,0,0,A,M,0,0,0,0,A,M,0,0,0],
    [0,0,S,0,0,0,0,0,0,0,0,S,0,0],
    [0,D,0,0,0,0,A,M,0,0,0,0,D,0],
    [0,0,0,0,0,0,M,A,0,0,0,0,0,0],
    [0,0,0,0,0,S,0,0,S,0,0,0,0,0],
    [0,0,0,0,D,0,0,0,0,D,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]