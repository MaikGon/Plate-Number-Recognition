import numpy as np
import cv2
from math import *
from pathlib import Path
from skimage.measure import compare_ssim


# Sorting corners for "perspective" function -> [ul, ur, bl, br]
def sort_cornes(corns, img):
    sorted_arr = []
    image_corns = []
    upper_right = [img.shape[1], 0]
    upper_left = [0, 0]
    bottom_right = [img.shape[1], img.shape[0]]
    bottom_left = [0, img.shape[0]]
    image_corns.append(upper_left)
    image_corns.append(upper_right)
    image_corns.append(bottom_left)
    image_corns.append(bottom_right)
    order = []
    ord = 0

    # Calculate distance between number plate corners and the whole image corners
    # to find out which is which [ul, ur, bl, br]
    for ind, val in enumerate(image_corns):
        lowest_dist = sqrt(img.shape[1]**2 + img.shape[0]**2)
        for i, v in enumerate(corns):
            dist = sqrt((val[0] - v[0])**2 + (val[1] - v[1])**2)
            if dist < lowest_dist:
                lowest_dist = dist
                ord = i
        order.append(ord)

    for o in order:
        sorted_arr.append(corns[o])

    # Once we have a sorted array, we can crop the number plate from image
    return perspective(sorted_arr, img)


# Crop number plate from image
def perspective(arr, img, width=1000, height=250):
    p1 = np.float32([arr[0], arr[1], arr[2], arr[3]])
    p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    get_tf = cv2.getPerspectiveTransform(p1, p2)
    persp = cv2.warpPerspective(img, get_tf, (width, height))

    return persp


# helper function for divide_tab function
def div_func(image, th=115):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, th, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    boxes = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # Check if the height and width is reasonable
        if h > 150 and (w > 15 and w < 200):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 1)
            boxes.append([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    return boxes


# Divide cropped area on segments with numbers and letters
def divide_tab(image, th=115):
    temp_img = image.copy()
    bboxes = div_func(image, th)

    if len(bboxes) < 7:
        th += 10
        if th >= 30:
            bboxes = div_func(temp_img, th)

    bboxes.sort()
    num_let_arr = []

    for b in bboxes:
        # cropped single values
        num_let = perspective(b, image, width=120, height=220)
        num_let_arr.append(num_let)

    # Return array with letters and numbers
    return num_let_arr


# Recognize each letter and number
def recognize(letters, im_paths):

    reference = []
    names = []
    # Read a reference images and add them to array (add also the names)
    for image_path in im_paths:
        image = cv2.imread(str(image_path), 0)
        reference.append(image)
        names.append(image_path.name[:1])

    text_arr = []

    # For each letter, calculate probability regarding the reference image (all letters and numbers)
    for let in letters:
        probabilites = []
        letters_arr = []
        img_gray = cv2.cvtColor(let, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret, thresh = cv2.threshold(blurred, 135, 255, cv2.THRESH_BINARY)

        for ind, im in enumerate(reference):
            im = cv2.resize(im, (thresh.shape[1], thresh.shape[0]))
            result, _ = compare_ssim(thresh, im, full=True)
            letter = names[ind]
            probabilites.append(result)
            letters_arr.append(letter)

        # Find letter which has the biggest probability and append it to array
        max_val = max(probabilites)

        for ind, val in enumerate(probabilites):
            if max_val == val:
                if letters_arr[ind] != 'w' and letters_arr[ind] != 'c' and letters_arr[ind] != 'r':
                    text_arr.append(letters_arr[ind])
    return ''.join(text_arr)


# helper function for perform_processing function
def help_perform(image, th=135, wind=3):
    wrapped_tab = None
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w = image.shape[1]
    h = image.shape[0]
    blurred = cv2.GaussianBlur(img_gray, (wind, wind), 3)
    _, thresh = cv2.threshold(blurred, th, 255, cv2.THRESH_BINARY_INV)

    cont, _ = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    hull_list = []

    for i in range(len(cont)):
        h = cv2.convexHull(cont[i])
        hull_list.append(h)

    contours = sorted(hull_list, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        clos = cv2.arcLength(c, True)
        apr = cv2.approxPolyDP(c, 0.05 * clos, True)
        cnt = 0
        nums = []
        points_arr = []

        # Check if c has 4 corners
        if len(apr) == 4:
            for i in range(4):
                j = i + 1
                if j == 4:
                    j = 0
                k = i - 1
                if k == -1:
                    k = 3

                # Calculate distance between corners and divide those values to find out if this is a plate number
                w_len_1 = sqrt((apr[i][0][0] - apr[j][0][0]) ** 2 + (apr[i][0][1] - apr[j][0][1]) ** 2)
                w_len_2 = sqrt((apr[i][0][0] - apr[k][0][0]) ** 2 + (apr[i][0][1] - apr[k][0][1]) ** 2)

                if w_len_1 > w_len_2:
                    if w_len_1 / w_len_2 < 3 or w_len_1 / w_len_2 > 7.3 or w_len_1 >= w:
                        continue
                else:
                    if w_len_2 / w_len_1 < 3 or w_len_2 / w_len_1 > 7.3 or w_len_2 >= w:
                        continue

                # Next condition to find out if this is a plate number
                if w_len_1 >= w / 3 or w_len_2 >= w / 3:
                    cnt += 1

                    points_arr.append(apr[i][0])
                if cnt == 2:
                    nums.append(apr)

        if len(nums) == 1:
            wrapped_tab = sort_cornes(points_arr, image)

    return wrapped_tab


# main function -> read image, process, etc.
def perform_processing(image: np.ndarray, ref) -> str:
    th = 135
    win = 3
    wrapped_tab = None
    for i in range(5):
        wrapped_tab = help_perform(image, th=th, wind=win)
        th -= 5
        win += 6
        if wrapped_tab is not None:
            break
    # if nothing was found, return a question marks
    if wrapped_tab is None:
        return '???????'
    try:
        single = divide_tab(wrapped_tab, th=115)
        resu = recognize(single, ref)
        if len(resu) == 7:
            return resu
        elif len(resu) > 7:
            x = len(resu) - 7
            return resu[x:]
        elif len(resu) == 0:
            return '???????'
        elif 7 > len(resu) > 0:
            x = 7 - len(resu)
            for i in range(x):
                resu = '?' + resu[0:]
            return resu
    except NameError:
        return '???????'

    return '???????'
