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


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image


def get_document_bounds(image_file, feature):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            bounds.append(symbol.bounding_box)

                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)

                if (feature == FeatureType.PARA):
                    bounds.append(paragraph.bounding_box)

            if (feature == FeatureType.BLOCK):
                bounds.append(block.bounding_box)

        if (feature == FeatureType.PAGE):
            bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds


def render_doc_text_google(filein, fileout):
    image = Image.open(filein)
    bounds = get_document_bounds(filein, FeatureType.PAGE)
    draw_boxes(image, bounds, 'blue')
    bounds = get_document_bounds(filein, FeatureType.PARA)
    draw_boxes(image, bounds, 'red')
    bounds = get_document_bounds(filein, FeatureType.WORD)
    draw_boxes(image, bounds, 'yellow')

    if fileout is not 0:
        image.save(fileout)
    else:
        image.show()


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


def extract_bill(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 31)
    # canny = cv2.Canny(gray, 50, 50)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    max_contour = find_max_contour(thresh)

    # coords = np.column_stack(np.where(thresh > 0))
    rect = cv2.minAreaRect(max_contour)
    angle = rect[-1]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_thresh = cv2.warpAffine(thresh, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_gray = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_thresh = cv2.copyMakeBorder(rotated_thresh, 2, 2, 0, 0, cv2.BORDER_CONSTANT, value=0)

    max_contour = find_max_contour(rotated_thresh)
    (x, y, w, h) = cv2.boundingRect(max_contour)
    gray = rotated_gray[int(y):int(y+h), int(x):int(x+w)]
    show_image(gray)

def find_max_contour(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    max_cnt_area = 0
    for c in contours:
        c_area = math.fabs(cv2.contourArea(c))
        if c_area > max_cnt_area:
            max_cnt_area = c_area
            max_contour = c
    return max_contour


def show_contours(img, contours):
    if img.ndim != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        img = cv2.drawContours(img, [contour], 0, (0, 0, 255))
    show_image(img)

def show_image(image, name='image'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 600)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-detect_file', default='resources/images/IMG_4099.JPG', help='The image for text detection.')
    parser.add_argument('-out_file', help='Optional output file', default=0)
    args = parser.parse_args()
    image = cv2.imread(args.detect_file)
    extract_bill(image)
    # render_doc_text_tesseract(args.detect_file)
    # render_doc_text(args.detect_file, args.out_file)


# # Instantiates a client
# client = vision.ImageAnnotatorClient()
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--image', required=False, default='resources/images/IMG_4063.JPG')
# args = parser.parse_args()
#
# image = cv2.imread(args.image)
# image = cv2.pyrDown(image)
# print(image.shape)
#
# # Loads the image into memory
# with io.open('resources/images/IMG_4063.JPG', 'rb') as image_file:
#     content = image_file.read()
#
# image = types.Image(content=content)
#
# response = client.document_text_detection(image=image)
# document = response.full_text_annotation
# print('Labels:')
# for label in labels:
#     print(label.description)
#     print(label)

# text = pytesseract.image_to_string(image, lang='ukr')
# print(text)