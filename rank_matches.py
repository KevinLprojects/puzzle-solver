from piece_registration import Match
from filter_matches import align_and_combine, display_match
from feature_detection import odd_size

import numpy as np
import cv2 as cv
import sys
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

import sys
sys.setrecursionlimit(3000)

EDGE_DILATION_PERCENT = 0.15
EDGE_BLUR_PERCENT = 0.3
CONTOUR_BLUR_PERCENT = 0.35

def edge_weight(maskA, maskB, piece_size):
    size = odd_size(np.sqrt(piece_size) * EDGE_DILATION_PERCENT)
    kernel = np.ones((size,size),np.uint8)
    maskA = cv.dilate(maskA, kernel, iterations = 1)
    maskB = cv.dilate(maskB, kernel, iterations = 1)

    edge_mask = np.logical_and(maskA, maskB)

    size = odd_size(np.sqrt(piece_size) * EDGE_BLUR_PERCENT)

    edge_mask_blurred = cv.GaussianBlur(edge_mask.astype(np.float32), (size, size), 0) * 255

    return (edge_mask_blurred / np.sum(edge_mask_blurred))

def overlap(maskA, maskB, weight):
    return np.sum(np.logical_and(maskA, maskB) * weight)

def underlap(mask, weight, piece_size):
    size = odd_size(np.sqrt(piece_size) * CONTOUR_BLUR_PERCENT)
    mask_blur = cv.GaussianBlur(mask.copy().astype(np.float32), (size, size), 0) > 0.5
    contours, hierarchy = cv.findContours(mask_blur.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    out = np.zeros_like(mask).astype(np.uint8)
    out = cv.fillPoly(out, [contours[0]], 255) > 0

    return np.sum(np.logical_xor(out, mask) * weight)

def color_match(imageA_exstended, imageB_exstended, weight):
    imageA_exstended = cv.cvtColor(imageA_exstended, cv.COLOR_RGB2HSV)
    imageB_exstended = cv.cvtColor(imageB_exstended, cv.COLOR_RGB2HSV)
    return np.sum(np.abs(imageA_exstended.astype(np.float32) - imageB_exstended.astype(np.float32)) * np.stack((weight, weight, weight), axis=2) / 255)

def cost(match, piece_size, pbar=None):
    plug_piece, pA, vA, socket_piece, pB, vB = match.get_transform_info()

    (warpedA, warpedB), R, t, shift = align_and_combine(plug_piece.mask, pA, vA, socket_piece.mask, pB, vB, img=True, separate=True)

    weight = edge_weight(warpedA, warpedB, piece_size)

    overlap_score = overlap(warpedA, warpedB, weight)

    underlap_score = underlap((warpedA + warpedB) > 0, weight, piece_size)

    (warpedA, warpedB), R, t, shift = align_and_combine(plug_piece.image_extended, pA, vA, socket_piece.image_extended, pB, vB, img=True, separate=True)

    color_score = color_match(warpedA, warpedB, weight)

    match.overlap_score = overlap_score
    match.underlap_score = 3 * underlap_score
    match.color_score = color_score / 2
    total_score = overlap_score + 3 * underlap_score + color_score / 2
    match.total_score = total_score

    if pbar:
        pbar.update(1)

    return total_score

def sort_sockets(socket):
    return socket[-1][0].total_score
    
if __name__ == "__main__":
    with open('puzzle_mask.pkl', 'rb') as f:
        mask, piece_size = pickle.load(f)

    with open('puzzle_pieces.pkl', 'rb') as f:
        piece_list = pickle.load(f)

    with open('num_matches.pkl', 'rb') as f:
        num_matches = pickle.load(f)

    pbar = tqdm(total=num_matches, desc="sorting matches")
    
    socket_list = []
    for piece in piece_list:
        for socket in piece.sockets:
            socket_list.append(socket)
            socket[-1].sort(key=partial(cost, piece_size, pbar=pbar))
    
    with open('puzzle_pieces.pkl', 'wb') as f:
        pickle.dump(piece_list, f)

    socket_list = []
    for piece in piece_list:
        for socket in piece.sockets:
            socket_list.append(socket)

    socket_list.sort(key=sort_sockets)

    image = Image.open('100_piece_puzzle.jpg').convert("RGB")
    image = np.array(image)

    for socket in socket_list:
        match = socket[-1][0]
        arrow_image = cv.arrowedLine(image.copy(), np.array([match.plug_piece.x, match.plug_piece.y]) + np.array(match.plug[0]), np.array([match.socket_piece.x, match.socket_piece.y]) + np.array(match.socket[0]), (0, 255, 0), 10, tipLength=0.1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) 
        ax1.imshow(arrow_image)
        ax1.axis('off')
        ax2.imshow(display_match(match))
        ax2.axis('off')
        plt.tight_layout()
        plt.show()