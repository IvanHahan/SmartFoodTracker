import numpy as np
from . import *
from .geometry import merge_intersections
from utils.util import *
import math


def put_boxes(img, boxes, color=(0, 0, 255), thickness=3):
    for rect in boxes:
        cv2.rectangle(img, (rect.x1, rect.y1), (rect.x2, rect.y2), color, thickness)
    return img


def get_image_hash(img, hash_size=8):
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def get_signature_for_box(image, box):
    return get_image_hash(image[box.y1:box.y2, box.x1:box.x2])


def paint_dominant(image, colors=2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    data = np.reshape(image, (-1)).astype('float32')
    ret, label, center = cv2.kmeans(data, colors, None, criteria, 10, flags)
    center = np.uint8(center)
    label = label.flatten()
    res = center[label]
    res2 = res.reshape((image.shape))
    return res2, center


def preprocess_for_ocr(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.average(img) < 170:
        img = adjust_gamma(img, 1.3)
    img = cv2.resize(img, (int(img.shape[1] * 1.5), int(img.shape[0] * 1.5)))

    img = sharpen(img, 2.5, 10)




    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if not dark_on_light(img):
        img = cv2.bitwise_not(img)

    return img


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def dark_on_light(img):
    return np.average(img) > 170


def remove_contours(img, contours, low_bound=300, up_bound=10000000, content_dark_on_light=False):
    mask = np.ones(img.shape, dtype="uint8") * 255
    reduced_contours = []
    if content_dark_on_light:
        img = cv2.bitwise_not(img)
    for c in contours:
        c_len = cv2.arcLength(c, True)
        if c_len < low_bound or c_len > up_bound:
            cv2.drawContours(mask, [c], -1, 0, -1)
        else:
            reduced_contours.append(c)
    img = cv2.bitwise_and(img, img, mask=mask)
    if content_dark_on_light:
        img = cv2.bitwise_not(img)
    return reduced_contours, img


def pyr_down(img, iters=1):
    for i in range(iters):
        img = cv2.pyrDown(img)
    return img


def pyr_up(img, iters=1):
    for i in range(iters):
        img = cv2.pyrUp(img)
    return img


def template_matches(img, template, threshold=0.7):
    iw, ih = img.shape
    w, h = template.shape[::-1]
    if iw < w or ih < h:
        return []
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    loc = list(zip(*loc[::-1]))
    boxes = [Box(pt[0], pt[1], pt[0] + w, pt[1] + h) for pt in loc]
    return boxes


def find_max_template_match(image, template):
    assert template.shape[0] < image.shape[0] and template.shape[1] < image.shape[1], \
        'Template should fit in image, template shape: {}, image shape: {}'.format(template.shape, image.shape)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, edge_max_loc = cv2.minMaxLoc(res)
    top_left = edge_max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    return Box(top_left[0], top_left[1], bottom_right[0], bottom_right[1]), max_val


def sharpen(img, blend_amount=0.5, sigma=10):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    diff = cv2.subtract(img, blurred)
    sharpened = cv2.addWeighted(img, 1, diff, blend_amount, 0)
    return sharpened


def dominant_colors(image, k=3):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_PP_CENTERS

    # Apply KMeans
    data = np.reshape(image, (-1, 1)).astype('float32')
    ret, label, name_centroids = cv2.kmeans(data, k, None, criteria, 10, flags)

    return name_centroids[np.argsort(np.bincount(label.flatten()))]


def extract_background(image):
    return dominant_colors(image, 5)[-1]


def gradient_boxes(image, axis=0):
    diff = np.diff(image, axis=axis, prepend=0)
    mask = np.bitwise_and(0 < diff, diff < 5).astype('uint8')
    gradient_mask = np.ones(diff.shape).astype('uint8') * 255
    gradient_mask = cv2.bitwise_and(gradient_mask, gradient_mask, mask=mask)
    gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    gradient_mask = cv2.copyMakeBorder(gradient_mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    contours = find_contours(gradient_mask)
    boxes = get_contour_boxes(contours, 100)
    return boxes


def paint_dominant(image, colors=2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    data = np.reshape(image, (-1, 1)).astype('float32')
    ret, label, name_centroids = cv2.kmeans(data, 2, None, criteria, 10, flags)
    painted = np.reshape(label, image.shape)
    show(painted)


def put_controls(image, controls):
    for control in controls:
        if control.control_type == Control.Type.BUTTON:
            image = put_boxes(image, [control.box], (0, 0, 255))
        elif control.control_type == Control.Type.TEXT_FIELD:
            # b = control.box
            # show(image[b.y1:b.y2, b.x1:b.x2])
            image = put_boxes(image, [control.box], (20, 190, 90))
        elif control.control_type == Control.Type.RADIO_BUTTON:
            image = put_boxes(image, [control.box], (190, 90, 0))
        else:
            image = put_boxes(image, [control.box], (90, 100, 80))
    return image


def get_active_boxes(left, right):
    if left.shape != right.shape:
        return None
    subtracted = cv2.subtract(right, left)
    thresh = cv2.threshold(subtracted, 0, 255, cv2.THRESH_BINARY)[1]
    dilate = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours = find_contours(dilate)
    filtered_contours = list(filter(lambda c: cv2.arcLength(c, True) > 100, contours))
    boxes = get_contour_boxes(filtered_contours)
    merge_intersections(boxes)
    return boxes, dilate


def get_boundaries(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(image, 20, 20)
    # canny = cv2.bitwise_and(canny, cv2.bitwise_not(sobel))
    contours = find_contours(canny)
    # rects = [cv2.boundingRect(c) for c in contours]
    boxes = get_contour_boxes(contours, 0)

    def remove_small_contours(canny, box_filter):

        contours = find_contours(canny)

        boxes = get_contour_boxes(contours, 0)

        boxes_to_remove = list(filter(box_filter, boxes))

        mask = np.ones(image.shape, dtype="uint8") * 255
        for c in boxes_to_remove:
            cv2.rectangle(mask, (c.x1, c.y1), (c.x2, c.y2), 0, -1)

        canny = cv2.bitwise_and(canny, canny, mask=mask)

        # canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        return canny

    canny = remove_small_contours(canny, lambda box: box.area < 400 and 0.2 < box.width / box.height < 1.8)
    canny = remove_small_contours(canny, lambda box: box.area < 400 and 0.2 < box.width / box.height < 1.8)
    canny = remove_small_contours(canny, lambda box: box.area < 400 and 0.2 < box.width / box.height < 1.8)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)

    canny = cv2.copyMakeBorder(canny, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    return canny


def remove_contours_on_condition(canny, condition):
    contours = find_contours(canny)

    boxes = get_contour_boxes(contours, 0)

    boxes_to_remove = list(filter(condition, boxes))

    mask = np.ones(canny.shape, dtype="uint8") * 255
    for c in boxes_to_remove:
        cv2.rectangle(mask, (c.x1, c.y1), (c.x2, c.y2), 0, -1)

    canny = cv2.bitwise_and(canny, canny, mask=mask)

    # canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    return canny


def orient_image(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    max_contour = find_max_contour(thresh)

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
    rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def find_max_contour(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    max_cnt_area = 0
    for c in contours:
        c_area = math.fabs(cv2.contourArea(c))
        if c_area > max_cnt_area:
            max_cnt_area = c_area
            max_contour = c
    return max_contour