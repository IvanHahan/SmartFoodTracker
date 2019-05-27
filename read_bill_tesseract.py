import argparse
from enum import Enum
import io

from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
import cv2
import pytesseract
import math
import numpy as np
from sklearn.cluster import KMeans
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from utils import util, geometry, image_processing
import ast
import re


def text_boxes(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] * image.shape[1] == 0:
        return []

    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9)))
    tophat = cv2.medianBlur(tophat, 11)
    tophat = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))

    thresh = cv2.threshold(tophat, 60, 255, cv2.THRESH_BINARY)[1]

    contours = image_processing.find_contours(thresh)

    return image_processing.get_contour_boxes(contours, 50)


def extract_groceries(filein):
    image = cv2.imread(filein)
    image = extract_bill(image)
    words = text_boxes(image)
    words = [w.inset(-5, -5, 5, 5) for w in words]
    # words = get_document_bounds(filein, FeatureType.WORD)

    geometry.merge_on(words, lambda l_b, r_b:  l_b.alignment_y(r_b) < 40 and l_b.is_horizontally_near_or_overlapped(r_b, 2.5))
    util.show(image, words)
    prices = []
    for word in words:
        word = word.inset(-5, -2, 5, 2)
        segment = image[word.y1:word.y2, word.x1:word.x2]
        if segment.shape[0] * segment.shape[1] == 0:
            continue
        preprocessed = image_processing.preprocess_for_ocr(segment)
        preprocessed = cv2.copyMakeBorder(preprocessed, 5, 5, 20, 20, cv2.BORDER_CONSTANT, value=255)
        preprocessed = image_processing.orient_image(preprocessed)
        text = pytesseract.image_to_string(preprocessed, lang='ukr')
        print(text)
        util.show(preprocessed)
        if re.match(r'\d{1,4}.\d{2} [А-ЯA-ZА-Я]', text):
            prices.append(word)


    util.show(image, prices)


def extract_bill(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 21)

    clahe = cv2.createCLAHE(2, (8, 8))
    blur = clahe.apply(blur).astype('uint8')

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    max_contour = image_processing.find_max_contour(thresh)

    # coords = np.column_stack(np.where(thresh > 0))
    rect = cv2.minAreaRect(max_contour)

    angle = rect[-1]
    if angle > 45:
        angle = 90 - angle
    elif angle < -45:
        angle = -90 - angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_thresh = cv2.warpAffine(thresh, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_gray = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_thresh = cv2.copyMakeBorder(rotated_thresh, 2, 2, 0, 0, cv2.BORDER_CONSTANT, value=0)

    max_contour = image_processing.find_max_contour(rotated_thresh)
    (x, y, w, h) = cv2.boundingRect(max_contour)
    gray = rotated_gray[int(y):int(y+h), int(x):int(x+w)]
    return gray


def render_doc_text_tesseract(file):
    image = cv2.imread(file)
    d = pytesseract.image_to_data(image, lang='ukr', output_type='dict')
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.namedWindow(file, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(file, 800, 600)
    cv2.imshow(file, image)
    cv2.waitKey(0)
