import time
import cv2
from .geometry import Box
from . import *
import numpy as np


def measure(original_function):
    def new_function(*args, **kwargs):
        start = time.time()
        value = original_function(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        print("{} elapsed: {}".format(original_function.__name__, elapsed_time))
        return value, elapsed_time, original_function.__name__

    return new_function


def show_boxes(img, boxes):
    if img.ndim != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for rect in boxes:
        img = cv2.rectangle(img, (rect.x1, rect.y1), (rect.x2, rect.y2), (0, 0, 255), 2)
    show(img)


def show(img, name='image'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 400)
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyWindow(name)


def show_contours(img, contours):
    if img.ndim != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        img = cv2.drawContours(img, [contour], 0, (0, 0, 255))
    show(img)


def find_contours(image):
    if cv2.__version__.split('.')[0] == '4':
        return cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    return cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]


def show_ocr_processing(image, name):
    if DEBUG_OCR_PREPROCESSING:
        show(image, name)


def show_top_bars(image):
    if DEBUG_TOP_BARS:
        show(image)


def show_text_boxes(image):
    if DEBUG_TEXT_BOXES:
        show(image)


def show_control_boxes(image):
    if DEBUG_CONTROL_BOXES:
        show(image)


def get_contour_boxes(contours, threshold_area=500.0):
    Rect = [cv2.boundingRect(i) for i in contours]
    RectP = [(int(i[0]), int(i[1]), int(i[0] + i[2]), int(i[1] + i[3])) for i in Rect]
    mapped_boxes = [Box(arr_box[0], arr_box[1], arr_box[2], arr_box[3]) for arr_box in RectP]
    expand = 0
    ex_boxes = [Box(b.x1 - expand, b.y1 - expand, b.x2 + expand, b.y2 + expand) for b in mapped_boxes]
    norm_boxes = [b for b in ex_boxes if b.area > threshold_area]
    return norm_boxes


def enter_values(prompt, valid_values, type_):
    try:
        values = type_(input(prompt))
        assert values in valid_values, 'Invalid value entered, try again'
        return values
    except Exception as e:
        print(e)
        return enter_values(prompt, valid_values, type_)
