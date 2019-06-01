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

urk_alphabet = 'йцукенгшщзхїґфівапролджєячсмитьбюёЙЦУКЕНГШЩЗХЇҐФІВАПРОЛДЖЄЯЧСМИТЬБЮЁ'

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


def extract_groceries(image_file):
    client = vision.ImageAnnotatorClient()

    blocks = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    img_content = types.Image(content=content)

    response = client.document_text_detection(image=img_content)
    document = response.full_text_annotation
    blocks = []
    for page in document.pages:
        for block in page.blocks:
            text = []
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        text.append(symbol.text)
                    text.append(' ')
                text.append('\n')
            text.append('\n')
            text = ''.join(text)
            blocks.append(text)

    print(blocks)

    for block in blocks:
        price = re.compile(r'\d{1,4}\s{0,1}\.\s{0,1}\d{2} [А-ЯA-ZА-Я]')
        if price.search(block) is not None:
            block = price.sub('\n', block)
            block = re.sub(r'Чек #', '', block)
            block = re.sub(r'\(.*\)', '', block)
            block = re.sub(r'[^{}\s]'.format(urk_alphabet), '', block)
            block = re.sub(r'\b[\S]{1,3}\b', '', block)
            block = re.sub(r'Знижка', '', block)
            block = re.sub(r'\ {2,}', ' ', block).strip()
            block = re.sub(r'\s{2,}', '\n', block)

            with open('out.txt', 'w') as out:
                out.write(block)
                break
        # bounds.append(annot.bounding_poly)

    # draw_boxes(image, bounds, 'red')
    # for page in document.pages:
    #     for block in page.blocks:
    #         for paragraph in block.paragraphs:
    #             for word in paragraph.words:
    #                 for symbol in word.symbols:




def render_doc_text_google(filein, fileout):
    image = Image.open(filein)
    # bounds = get_document_bounds(filein, FeatureType.PAGE)
    # draw_boxes(image, bounds, 'blue')
    extract_groceries(filein)
    bounds = get_document_bounds(filein, FeatureType.BLOCK)

    # bounds = get_document_bounds(filein, FeatureType.WORD)
    draw_boxes(image, bounds, 'red')

    if fileout is not 0:
        image.save(fileout)
    else:
        image.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-detect_file', default='resources/images/IMG_4077.JPG', help='The image for text detection.')
    parser.add_argument('-out_file', help='Optional output file', default=0)
    args = parser.parse_args()
    image = cv2.imread(args.detect_file)
    render_doc_text_google(args.detect_file, args.out_file)
