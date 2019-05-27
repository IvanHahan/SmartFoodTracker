from math import fabs
import numpy as np
import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Box:
    def __init__(self, x1, y1, x2, y2, m=0, s=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.merged_times = m
        self.area_size_index = s
        self.text = ''

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center(self):
        return Point(int(self.x1 + self.width / 2), int(self.y1 + self.height / 2))

    def intersection(self, box):
        x1 = max(self.x1, box.x1)
        x2 = min(self.x2, box.x2)
        y1 = max(self.y1, box.y1)
        y2 = min(self.y2, box.y2)
        if x1 < x2 and y1 < y2:
            return Box(x1, y1, x2, y2)
        return Box(0,0,0,0)

    def union(self, box):
        x1 = min(self.x1, box.x1)
        x2 = max(self.x2, box.x2)
        y1 = min(self.y1, box.y1)
        y2 = max(self.y2, box.y2)
        b = Box(x1, y1, x2, y2)
        b.text = self.text + box.text
        return b

    def iou(self, box):
        return float(self.intersection(box).area) / (float(self.union(box).area) + 0.0001)

    def inset(self, x1, y1, x2, y2):
        return Box(self.x1 + x1, self.y1 + y1, self.x2 + x2, self.y2 + y2)

    def adjust(self, size):
        h, w = size
        self.x1 = self.x1 if self.x1 >= 0 else 0
        self.y1 = self.y1 if self.y1 >= 0 else 0
        self.x2 = self.x2 if self.x2 >= 0 else 0
        self.y2 = self.y2 if self.y2 >= 0 else 0

        self.x1 = self.x1 if self.x1 < w else w - 1
        self.y1 = self.y1 if self.y1 < h else h - 1
        self.x2 = self.x2 if self.x2 < w else w - 1
        self.y2 = self.y2 if self.y2 < h else h - 1
        return self

    def __contains__(self, item):
        return item.x1 > self.x1 and item.x2 < self.x2 \
               and item.y1 > self.y1 and item.y2 < self.y2

    def contains_point(self, point):
        return self.x1 <= point.x <= self.x2 \
               and self.y1 <= point.y <= self.y2

    def alignment_y(self, box):
        return fabs(self.y1 - box.y1) + fabs(self.y2 - box.y2)

    def is_vertically_near_or_overlapped(self, rect, distance_rate=0.02):
        h_1 = (self.y2 - self.y1) * distance_rate
        h_2 = (rect.y2 - rect.y1) * distance_rate
        return (rect.y1 - h_2 < self.y1 < rect.y2 + h_2) or \
               (rect.y1 - h_2 < self.y2 < rect.y2 + h_2) or \
               (self.y1 - h_1 < rect.y1 < self.y2 + h_1) or \
               (self.y1 - h_1 < rect.y1 < self.y2 + h_1)

    def __str__(self):
        return "x1:{} y1:{} x2:{} y2:{} area: {}".format(self.x1, self.y1, self.x2, self.y2, self.area)

    def __repr__(self):
        return "x1:{} y1:{} x2:{} y2:{} area: {}".format(self.x1, self.y1, self.x2, self.y2, self.area)

    @staticmethod
    def from_2_pos(x1, y1, x2, y2, m=0, s=0):
        return Box(x1, y1, x2 - x1, y2 - y1, m, s)

    def convert_to(self, box):
        return Box(self.x1+box.x1, self.y1+box.y1,
                   self.x2+box.x1, self.y2+box.y1)

    def is_rectangles_overlapped_horizontally(self, rect_2):
        return self.is_horizontally_near_or_overlapped(rect_2) and \
               self._is_height_of_rects_match(rect_2) and \
               (self.is_vertically_near_eachother(rect_2) or
                self._is_vertically_included(rect_2))

    def is_vertically_near_eachother(self, rect_2):
        distance_rate = 0.2
        h_1 = (self.y2 - self.y1) * distance_rate
        h_2 = (rect_2.y2 - rect_2.y1) * distance_rate
        return math.fabs(rect_2.y1 - self.y1) < h_1 or \
               math.fabs(rect_2.y2 - self.y2) < h_1 or \
               math.fabs(rect_2.y1 - self.y1) < h_2 or \
               math.fabs(rect_2.y2 - self.y2) < h_2

    def _is_height_of_rects_match(self, rect_2):
        max_vertical_diff_rate = 1.75
        return self.height * max_vertical_diff_rate > rect_2.height and \
               rect_2.height * max_vertical_diff_rate > self.height

    def _is_vertically_included(self, rect_2):
        return (rect_2.y1 < self.y1 and rect_2.y2 > self.y2) or \
               (self.y1 < rect_2.y1 and self.y2 > rect_2.y2)

    def is_horizontally_near_eachother(self, rect_2):
        distance_rate = 0.2
        w_1 = (self.x2 - self.x1) * distance_rate
        w_2 = (rect_2.x2 - rect_2.x1) * distance_rate
        return math.fabs(rect_2.x1 - self.x1) < w_1 or \
               math.fabs(rect_2.x2 - self.x2) < w_1 or \
               math.fabs(rect_2.x1 - self.x1) < w_2 or \
               math.fabs(rect_2.x2 - self.x2) < w_2

    def is_horizontally_near_or_overlapped(self, rect_2, horizontal_distance_rate=0.02):
        w_1 = (self.y2 - self.y1) * horizontal_distance_rate
        w_2 = (rect_2.y2 - rect_2.y1) * horizontal_distance_rate
        return (rect_2.x1 - w_2 < self.x1 < rect_2.x2 + w_2) or \
               (rect_2.x1 - w_2 < self.x2 < rect_2.x2 + w_2) or \
               (self.x1 - w_1 < rect_2.x1 < self.x2 + w_1) or \
               (self.x1 - w_1 < rect_2.x1 < self.x2 + w_1)


def merge_intersections(boxes, threshold=0):
    i = 0
    merge_happened = False
    while i < len(boxes):
        j = i + 1
        while j < len(boxes):
            iou = boxes[i].iou(boxes[j])
            if iou > threshold:
                boxes[i] = boxes[i].union(boxes[j])
                del boxes[j]
                merge_happened = True
            j += 1
        i += 1
    if merge_happened:
        merge_intersections(boxes, threshold)


def merge_on(boxes, condition):
    i = 0
    merge_happened = False
    while i < len(boxes):
        j = i + 1
        while j < len(boxes):
            fst = boxes[i]
            snd = boxes[j]
            if condition(fst, snd):
                boxes[i] = boxes[i].union(boxes[j])
                del boxes[j]
                merge_happened = True
            j += 1
        i += 1
    if merge_happened:
        merge_on(boxes, condition)


def average_intersections(boxes, threshold=0):
    i = 0
    merge_happened = False
    while i < len(boxes):
        j = i + 1
        while j < len(boxes):
            iou = boxes[i].iou(boxes[j])
            if iou > threshold:
                b1 = boxes[i]
                b2 = boxes[j]

                x1 = np.mean([b1.x1, b2.x1])
                y1 = np.mean([b1.y1, b2.y1])
                x2 = np.mean([b1.x2, b2.x2])
                y2 = np.mean([b1.y2, b2.y2])

                boxes[i] = Box(int(x1), int(y1), int(x2), int(y2))
                del boxes[j]
                merge_happened = True
            j += 1
        i += 1
    if merge_happened:
        average_intersections(boxes, threshold)


def merge_text_boxes(boxes):
    i = 0
    merge_happened = False
    while i < len(boxes):
        j = i + 1
        while j < len(boxes):
            lb = boxes[i]
            rb = boxes[j]
            if lb.is_rectangles_overlapped_horizontally(rb):
                boxes[i] = lb.union(rb)
                del boxes[j]
                merge_happened = True
            j += 1
        i += 1
    if merge_happened:
        merge_text_boxes(boxes)


def get_vectors(l, r):
    x1, y1, x2, y2 = l
    px1, py1, px2, py2 = r

    x2 = max(x2, x1) - min(x2, x1)
    y2 = max(y2, y1) - min(y2, y1)

    px2 = max(px2, px1) - min(px2, px1)
    py2 = max(py2, py1) - min(py2, py1)
    return (x2, y2), (px2, py2)


def get_cos(l, r):
    (x2, y2), (px2, py2) = get_vectors(l, r)

    cos = (x2*px2+y2*py2) / (math.sqrt(x2*x2 + y2*y2) * math.sqrt(px2*px2 + py2*py2))
    return cos


def get_distance(l, r):
    x1, y1, x2, y2 = l
    px1, py1, px2, py2 = r

    dx = px2 - x2
    dy = py2 - y2

    x1 = px1 - dx
    y1 = py1 - dy


    c1 = x2 * y1 - x1 * y2
    c2 = px2 * py1 - px1 * py2

    a = y2 - y1
    b = x1 - x2

    distance = math.fabs(c2 - c1) / ((math.sqrt(a * a + b * b))+0.001)
    return distance
