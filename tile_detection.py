import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import resize

from global_variables import *
#---------------------- for solving---------------------------------------

def print_table(board):
    for i in range(14):
        for j in range(14):
            print(f"{board[i][j]:2}", end=" ")  
        print()  

def find_move_coordonates(image1, image2):
    tile_differences = []
    indexes = [] 

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([80, 60, 255])

    image1 = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
    image1 = cv.inRange(image1, lower_bound, upper_bound)
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
    image2 = cv.inRange(image2, lower_bound, upper_bound)
    difference = image2 - image1

    # show_image(image1)
    # show_image(image2)
    # show_image(diferenta)
   
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20

            tile_difference = difference[x_min:x_max, y_min:y_max]
            diferenta_patrat = np.sum(tile_difference)
            tile_differences.append(diferenta_patrat)
            indexes.append((i, j))  
    sorted_pairs = sorted(zip(tile_differences, indexes), reverse=True)
    sorted_tile_differences, indexes_sorted = zip(*sorted_pairs)
    sorted_tile_differences = list(sorted_tile_differences)
    indexes_sorted = list(indexes_sorted)
    # print(diferente_sorted)
    # print(indexes_sorted)
    max_tile_diff = max(tile_differences)
    tile_move_coords = indexes[tile_differences.index(max_tile_diff)] 
    return tile_move_coords


def filter_tile(tile):
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 255, 150])
    tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    tile = cv.inRange(tile, lower_bound, upper_bound)
    kernel = np.ones((2, 2), np.uint8)
    tile = cv.erode(tile, kernel)
    tile_copy = cv.cvtColor(tile, cv.COLOR_GRAY2BGR)
    return tile, tile_copy

def normalize_digit(digit):
    height, width = digit.shape
    offset_height = (biggest_height-height)//2
    offset_width = (biggest_width-width)//2
    if offset_height >= 0 and offset_width >= 0:
        normalized_digit = np.zeros((biggest_height, biggest_width), dtype=np.uint8)
        normalized_digit[offset_height:offset_height+height,offset_width:offset_width+width] = digit
    else:
        normalized_digit = resize(digit, (biggest_height, biggest_width), mode='constant', anti_aliasing=True)
    return normalized_digit


def predict_class(image, kmeans):
    image = normalize_digit(image)
    image_flattened = image.flatten()

    cluster_label = kmeans.predict([image_flattened])[0]
    predicted_class = class_to_cluster_idx.index(cluster_label)
    return int(predicted_class)

def cut_tile_digits(kmeans, board,move_x, move_y):
    found_digits = []
    result = 0
    y_min = lines_vertical[move_y][0][0] 
    y_max = lines_vertical[move_y + 1][1][0] 
    x_min = lines_horizontal[move_x][0][1] 
    x_max = lines_horizontal[move_x + 1][1][1] 
    tile = board[x_min:x_max,y_min:y_max] 
    
    tile,tile_copy = filter_tile(tile)

    contours, _ = cv.findContours(tile,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if(len(contours[i]) >10):
            top_x = None
            bottom_x = None
            left_y = None
            right_y = None
            for point in contours[i].squeeze():
                if left_y is None or point[0] < left_y:
                    left_y = point[0]

                if right_y is None or point[0] > right_y:
                    right_y = point[0]
                
                if top_x is None or point[1] < top_x:
                    top_x = point[1]

                if bottom_x is None or point[1] > bottom_x:
                    bottom_x = point[1]

            area = cv.contourArea(np.array([(left_y,top_x),(left_y,bottom_x),(right_y,bottom_x),(right_y,top_x)]))
            width = right_y - left_y
            height = bottom_x - top_x
            # if(area > 700 and area < 2000):
            if width > 10 and width < 40 and height > 45 and height < 60:
                found_digit_image = cv.cvtColor(tile_copy[top_x:bottom_x,left_y:right_y],cv.COLOR_BGR2GRAY)
                found_digit = cut_digit(top_x, bottom_x, left_y, right_y,area, found_digit_image)
                found_digits.append(found_digit)
                # show_image(found_digit_image)

                cv.circle(tile_copy,(left_y, top_x),2,(0,0,255),-1)
                cv.circle(tile_copy,(right_y, top_x),2,(0,0,255),-1)
                cv.circle(tile_copy,(left_y, bottom_x),2,(0,0,255),-1)
                cv.circle(tile_copy,(right_y, bottom_x),2,(0,0,255),-1)

    if len(found_digits) == 1:
        result = predict_class(found_digits[0].image, kmeans)
    elif len(found_digits) == 2:
        if found_digits[0].left_y < found_digits[1].left_y:
            first_digit = predict_class(found_digits[0].image, kmeans)
            second_digit = predict_class(found_digits[1].image, kmeans)
        else:
            first_digit = predict_class(found_digits[1].image, kmeans)
            second_digit = predict_class(found_digits[0].image, kmeans)
        result = first_digit*10 + second_digit
    # show_image(tile)
    return result, tile_copy

def constrained_ecuation(x,y,n1,n2,n):
    score = 0
    if CONSTRAINTS[x][y] == A and n1 + n2 == n:
        # print("adunare", n1, n2)
        score = n
    elif CONSTRAINTS[x][y] == S and abs(n1 - n2) == n:
        # print("scadere", n1, n2)
        score = n
    elif CONSTRAINTS[x][y] == M and n1*n2 == n:
        # print("inmultire", n1, n2)
        score = n  
    elif int(min(n1,n2)) != 0 and CONSTRAINTS[x][y] == D and int(max(n1,n2))/int(min(n1,n2)) == int(n):
        # print("impartire", n1, n2)
        score = n
    return score

def uncontrained_ecuation(n1,n2,n):
    score = 0
    if n1 + n2 == n:
        # print("adunare", n1, n2)
        score = n
    elif abs(n1 - n2) == n:
        # print("scadere", n1, n2)
        score = n
    elif n1*n2 == n:
        # print("inmultire", n1, n2)
        score += n  
    elif int(min(n1,n2))!=0 and int(max(n1,n2))/int(min(n1,n2)) == int(n):
        # print("impartire", n1, n2)
        score = n
    return score

def ecuation(x,y,n1,n2,n):
    score = 0
    if CONSTRAINTS[x][y] != 0:
        # print("constraint")
        score += constrained_ecuation(x,y,n1,n2,n)
    else:
        # print("uncontrained")
        score += uncontrained_ecuation(n1,n2,n)
    return score

def calculate_move_score(x, y, n, board_pieces):
    # print(f"x= {x} y={y}")
    score = 0
    if x >= 2:
        n1 = board_pieces[x-2][y]
        n2 = board_pieces[x-1][y]
        if n1!=-1 and n2!=-1:
            # print("n1 si n2 pe sus",n1,n2)
            score += ecuation(x,y,n1,n2,n)

    if x <= 11:
        n1 = board_pieces[x+1][y]
        n2 = board_pieces[x+2][y]
        if n1!=-1 and n2!=-1:
            # print("n1 si n2 pe jos",n1,n2)
            score += ecuation(x,y,n1,n2,n)

    if y >= 2:
        n1 = board_pieces[x][y-2]
        n2 = board_pieces[x][y-1]

        if n1!=-1 and n2!=-1:
            # print("n1 si n2 pe stanga: ", n1, n2)
            score += ecuation(x,y,n1,n2,n)

    if y <= 11:
        n1 = board_pieces[x][y+1]
        n2 = board_pieces[x][y+2]
        if n1!=-1 and n2!=-1:
            # print("n1 si n2 pe dreapta: ", n1, n2)
            score += ecuation(x,y,n1,n2,n)

    # if BONUSES[x][y]:
        # print(f"bonus X{BONUSES[x][y]}")
    score = score * BONUSES[x][y]
    return score

def calculate_move(kmeans,game,empty_board, images, turns_text, output_dir):
    output_path = os.path.join(output_dir,f"{game+1}_scores.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        board_pieces = copy.deepcopy(BOARD_PIECES)
        players = []
        turns = []
        for line in turns_text.strip().splitlines():
            player, turn = line.split()
            players.append(player)
            turns.append(int(turn))

        turns_idx = 0
        my_turn_score = 0
        current_player = players[turns_idx][-1]
        first_player_turn = int(turns[turns_idx]) - 1
        last_player_turn = int(turns[turns_idx+1]) - 2
        for idx in range(0,50):
            if  idx > last_player_turn:
                my_turn_score = 0
                turns_idx += 1

                if turns_idx + 1 == len(turns):
                    current_player = players[turns_idx][-1]
                    first_player_turn = int(turns[turns_idx]) - 1
                    last_player_turn = 49
                else:
                    current_player = players[turns_idx][-1]
                    first_player_turn = int(turns[turns_idx]) - 1
                    last_player_turn = int(turns[turns_idx+1]) - 2
            if idx == 0:
                image1 = empty_board
                image2 = images[idx]

            else:
                image1 = images[idx-1]
                image2 = images[idx]
            indexes = find_move_coordonates(image1,image2)
            linie = indexes[0] + 1
            coloana =chr(indexes[1] + 65)
            # my_moved_tile_indexes = str(linie) + str(coloana)

            # data = txts[idx]
            # true_moved_tile_indexes = data.split(' ', 1)[0]
            # remaining_data = data.split(' ', 1)[1] if ' ' in data else ""
            # true_tile_number = int(remaining_data.split(' ', 1)[0])

            my_tile_number,processed_tile = cut_tile_digits(kmeans, images[idx],indexes[0],indexes[1])
            
            board_pieces[indexes[0]][indexes[1]] = my_tile_number

            score = calculate_move_score(indexes[0], indexes[1], my_tile_number,board_pieces)

            if idx < 9:
                output_path = os.path.join(output_dir,f"{game+1}_0{idx+1}.txt")
            else:
                output_path = os.path.join(output_dir,f"{game+1}_{idx+1}.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{linie}{coloana} {my_tile_number}")
            my_turn_score += score

            
            if idx == last_player_turn:
                file.write(f"Player{current_player} {first_player_turn + 1} {my_turn_score}\n")
    print_table(board_pieces)


#--------------------for image processing-----------------------------------



def cut_tiles(image):
    image_copy = image.copy()
    tiles = []
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([90, 60, 255])

    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = cv.inRange(image, lower_bound, upper_bound)
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20

            patrat = image[x_min:x_max, y_min:y_max]
            diferenta_patrat = np.sum(patrat)
            if diferenta_patrat > 400000:
                y_min = lines_vertical[j][0][0] 
                y_max = lines_vertical[j + 1][1][0]
                x_min = lines_horizontal[i][0][1] 
                x_max = lines_horizontal[i + 1][1][1] 
                # show_image(image_copy[x_min:x_max, y_min:y_max])
                tiles.append(image_copy[x_min:x_max, y_min:y_max])

    return tiles


def cut_digits(tiles, output_folder):
    image_idx = 0
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 255, 150])

    for tile_idx in range(len(tiles)):
        tile = tiles[tile_idx]
        tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        tile = cv.inRange(tile, lower_bound, upper_bound)
        tile_copy = cv.cvtColor(tile, cv.COLOR_GRAY2BGR)
        # tile, tile_copy = filter_tile(tile)
        contours, _ = cv.findContours(tile,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            if(len(contours[i]) >10):
                top_x = None
                bottom_x = None
                left_y = None
                right_y = None
                for point in contours[i].squeeze():
                    if top_x is None or point[0] < top_x:
                        top_x = point[0]

                    if bottom_x is None or point[0] > bottom_x:
                        bottom_x = point[0]
                    
                    if left_y is None or point[1] < left_y:
                        left_y = point[1]

                    if right_y is None or point[1] > right_y:
                        right_y = point[1]

                area = cv.contourArea(np.array([(top_x,left_y),(top_x,right_y),(bottom_x,right_y),(bottom_x,left_y)]))
                if(area > 800 and area < 2000):
                    # print(area)
                    # show_image(original_image[left_y:right_y,top_x:bottom_x])
                    image_idx += 1
                    cv.imwrite(os.path.join(output_folder,f"{image_idx}.jpg"), cv.cvtColor(tile_copy[left_y:right_y,top_x:bottom_x],cv.COLOR_BGR2GRAY))






def calculate_move_train(kmeans, game,empty_board, images, txts, turns_text, scores_txt):

    board_pieces = copy.deepcopy(BOARD_PIECES)
    players = []
    turns = []
    for line in turns_text.strip().splitlines():
        player, turn = line.split()
        players.append(player)
        turns.append(int(turn))
    scores = []
    for line in scores_txt.strip().splitlines():
        _,_,score  = line.split()
        scores.append(int(score))


    turns_idx = 0
    correct_turn_score = int(scores[turns_idx])
    my_turn_score = 0
    current_player = players[turns_idx][-1]
    last_player_turn = int(turns[turns_idx+1]) - 2
    for idx in range(0,50):
        
        if  idx > last_player_turn:
            my_turn_score = 0
            turns_idx += 1
            correct_turn_score = int(scores[turns_idx])

            if turns_idx + 1 == len(turns):
                last_player_turn = 50
                current_player = players[turns_idx][-1]
            else:
                current_player = players[turns_idx][-1]
                last_player_turn = int(turns[turns_idx+1]) - 2
        if idx == 0:
            image1 = empty_board
            image2 = images[idx]

        else:
            image1 = images[idx-1]
            image2 = images[idx]
        indexes = find_move_coordonates(image1,image2)
        linie = indexes[0] + 1
        coloana =chr(indexes[1] + 65)
        my_moved_tile_indexes = str(linie) + str(coloana)
        data = txts
        true_moved_tile_indexes = data.split(' ', 1)[0]
        remaining_data = data.split(' ', 1)[1] if ' ' in data else ""
        true_tile_number = int(remaining_data.split(' ', 1)[0])

        is_correct_tile = False
        if my_moved_tile_indexes == true_moved_tile_indexes:
            is_correct_tile = True
        my_tile_number,processed_tile = cut_tile_digits(kmeans, images[idx],indexes[0],indexes[1])
        is_correct_number = False
        if my_tile_number == true_tile_number:
            is_correct_number = True
        if is_correct_number == False:
            print("false nr detected")
        board_pieces[indexes[0]][indexes[1]] = my_tile_number

        score = calculate_move_score(indexes[0], indexes[1], my_tile_number,board_pieces)
        my_turn_score += score
        if idx == last_player_turn:
            is_correct_score = True
            if correct_turn_score != my_turn_score:
                is_correct_score = False
            print(f"{is_correct_score} correct score : {correct_turn_score} my score: {my_turn_score}")

        # print(f"current player: {current_player}")
        # print(f"runda {game+1} mutare {idx+1}")
        # print(f"detectat: {my_tile_number}")
        # print(f"scor mutare: {score}")
        # show_image(processed_tile)
        # if idx == 14:
        #     print_table(board_pieces)
        # print("\n")



