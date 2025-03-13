import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

from global_variables import *



#----------------------for resolving----------------------------------------------------------------
def extract_big_board(image):

    image_m_blur = cv.medianBlur(image,11)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 2, image_g_blur, -0.8, 0)
    # show_image(image_sharpened)

    hsv_image = cv.cvtColor(image_sharpened, cv.COLOR_BGR2HSV)

    lower_bound = np.array([20, 0, 150])
    upper_bound = np.array([255, 255, 255])

    thresh = cv.inRange(hsv_image, lower_bound, upper_bound)


    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel)
    # show_image(thresh)

    edges =  cv.Canny(thresh ,200,400)
    # show_image(edges)

    contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    # print(top_left)
    # print(bottom_right)
    # print(bottom_left)
    # print(top_right)

    width = 2000
    height = 2000
    
    # image_copy = image.copy()
    # cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    # show_image(image_copy)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

    M = cv.getPerspectiveTransform(puzzle,destination)
    result = cv.warpPerspective(image , M,(width, height))
    
    return result


def extract_all_big_boards(input_folder):
    input_files = os.listdir(input_folder)
    input_files = sorted(input_files)

    big_boards = []

    for file in input_files:
        if file[-3:] == "jpg":
            image = cv.imread(os.path.join(input_folder, file))
            big_board = extract_big_board(image)
            big_boards.append(big_board)
    return big_boards

def extract_all_boards_by_average_coordonates(big_boards):
    boards = []
    for board in big_boards:
        new_board = extract_board_by_average_coordonates(board)
        boards.append(new_board)
    return boards


#--------------------------------------for training processing------------------------------------
def extract_board(image):
    boxes = []
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_bound = np.array([15, 15, 0])
    upper_bound = np.array([90, 255, 255])

    thresh = cv.inRange(hsv_image, lower_bound, upper_bound)

    image_m_blur = cv.medianBlur(thresh, 3)
    # print("median blur")
    # show_image(image_m_blur)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    # print("gaussian blur")
    # show_image(image_g_blur)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    # print("sharpening")
    # show_image(image_sharpened)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(image_sharpened, kernel)
    # print("thresh")
    # show_image(thresh)

    edges = cv.Canny(thresh, 200, 400)
    # print("edges")
    # show_image(edges)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if len(contours[i]) > 3:
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]

            area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
            if area > 7000:
                boxes.append(
                        box(possible_top_left, possible_top_right, possible_bottom_left, possible_bottom_right, area)
                    )

    # image_copy = image.copy()
    # for idx, curr_box in enumerate(boxes):
    #     cv.circle(image_copy, tuple(curr_box.top_left), 5, (255, 0, 0), -1) 
    #     cv.circle(image_copy, tuple(curr_box.top_right), 5, (0, 255, 0), -1) 
    #     cv.circle(image_copy, tuple(curr_box.bottom_left), 5, (0, 0, 255), -1)  
    #     cv.circle(image_copy, tuple(curr_box.bottom_right), 5, (255, 0, 255), -1)  
    # show_image(image_copy)

    # boxes.sort(key=lambda b: b.area, reverse=True)

    # for idx, b in enumerate(boxes):
    #     print(b.area)

        
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    for idx, curr_box in enumerate(boxes):
        if top_left is None or curr_box.top_left[0] + curr_box.top_left[1] < top_left[0] + top_left[1]:
            top_left = curr_box.top_left.copy()

        if bottom_right is None or curr_box.bottom_right[0] + curr_box.bottom_right[1] > bottom_right[0] + bottom_right[1]:
            bottom_right = curr_box.bottom_right.copy()

        if  top_right is None or curr_box.top_right[0] - curr_box.top_right[1] > top_right[0] - top_right[1]:
            top_right = curr_box.top_right.copy()

        if  bottom_left is None or curr_box.bottom_left[0] - curr_box.bottom_left[1] < bottom_left[0] - bottom_left[1]:
            bottom_left = curr_box.bottom_left.copy()

    if top_left is None or top_right is None or bottom_left is None or bottom_right is None:
        return None, None

    

    width = 1260
    height = 1260
        
    # image_copy = image.copy()
    # cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    # cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    # show_image(image_copy)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

    M = cv.getPerspectiveTransform(puzzle,destination)
    result = cv.warpPerspective(image , M,(width, height))

    area = cv.contourArea(np.array([[top_left],[top_right],[bottom_right],[bottom_left]]))
    careu_values = box(top_left, top_right, bottom_left, bottom_right, area)
              
    return careu_values, result
        
    


def find_corner_coordonates(corner,careu_values_array):

    x_values = [getattr(cur_careu, corner)[0] for cur_careu in careu_values_array]
    value_counts_x = Counter(x_values)
    most_common_x = max(value_counts_x.items(), key=lambda x: x[1])
    most_common_value_x = most_common_x[0]

    y_values = [getattr(cur_careu, corner)[1] for cur_careu in careu_values_array]
    value_counts_y = Counter(y_values)
    most_common_y = max(value_counts_y.items(), key=lambda x: x[1])
    most_common_value_y = most_common_y[0]

    return most_common_value_x, most_common_value_y


def extract_board_by_average_coordonates(image):  
    width = 1400
    height = 1400
    careu = box((268,263),(1734,269),(262,1739),(1736,1749),0)     
    puzzle = np.array([[careu.top_left[0],careu.top_left[1]], [careu.top_right[0],careu.top_right[1]]
                        ,[careu.bottom_right[0],careu.bottom_right[1]], [careu.bottom_left[0],careu.bottom_left[1]]], dtype='float32')
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    M = cv.getPerspectiveTransform(puzzle,destination)
    result = cv.warpPerspective(image , M,(width, height))
    return result