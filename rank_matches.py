"""
Course Number: ENGR 13300
Semester: e.g. Spring 2025

Description:
    Uses various match evaluation metrics to sort matches for each socket.

Assignment Information:
    Assignment:     18 Ind Project
    Team ID:        LC2 - 11
    Author:         Kevin LeRoy, leroyk@purdue.edu
    Date:           12/08/2025

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

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

# pickle was complaining
import sys
sys.setrecursionlimit(3000)

EDGE_DILATION_PERCENT = 0.15 # percent of sqrt piece size to use for dilation in edge overlap
EDGE_BLUR_PERCENT = 0.3 # percent of sqrt piece size to blur edge overlap
CONTOUR_BLUR_PERCENT = 0.35 # percent of sqrt piece size to blur mask to check for smooth combined piece contour

# weight for cost based (higher close to the shared edge sum=1)
def edge_weight(maskA, maskB, piece_size):
    # dilate the masks to find the overlap
    size = odd_size(np.sqrt(piece_size) * EDGE_DILATION_PERCENT)
    kernel = np.ones((size,size),np.uint8)
    maskA = cv.dilate(maskA, kernel, iterations = 1)
    maskB = cv.dilate(maskB, kernel, iterations = 1)

    edge_mask = np.logical_and(maskA, maskB)

    size = odd_size(np.sqrt(piece_size) * EDGE_BLUR_PERCENT)

    # blur the mask the weight closer to the shared edge higher
    edge_mask_blurred = cv.GaussianBlur(edge_mask.astype(np.float32), (size, size), 0) * 255

    # normalize
    return (edge_mask_blurred / np.sum(edge_mask_blurred))

# get the overlap of two masks
def overlap(maskA, maskB, weight):
    return np.sum(np.logical_and(maskA, maskB) * weight)

# get the "underlap" of two masks
def underlap(mask, weight, piece_size):
    # blur the mask to check if the composited masks have smooth edges
    size = odd_size(np.sqrt(piece_size) * CONTOUR_BLUR_PERCENT)
    mask_blur = cv.GaussianBlur(mask.copy().astype(np.float32), (size, size), 0) > 0.5
    contours, hierarchy = cv.findContours(mask_blur.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # fill the shared piece contour
    out = np.zeros_like(mask).astype(np.uint8)
    out = cv.fillPoly(out, [contours[0]], 255) > 0

    # get the normalized sum of underlapping areas
    return np.sum(np.logical_xor(out, mask) * weight)

# how the colors match
def color_match(imageA_exstended, imageB_exstended, weight):
    # convert to hsv, where the color diff is more meaningfull (there is probably a better color space and dist func for this)
    imageA_exstended = cv.cvtColor(imageA_exstended, cv.COLOR_RGB2HSV)
    imageB_exstended = cv.cvtColor(imageB_exstended, cv.COLOR_RGB2HSV)

    # git diff of colors normalized by the edge weight
    return np.sum(np.abs(imageA_exstended.astype(np.float32) - imageB_exstended.astype(np.float32)) * np.stack((weight, weight, weight), axis=2) / 255)

# combine all the scores for a given mask
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

# used for sorting the sockets by their best match's score
def sort_sockets(socket):
    return socket[-1][0].total_score

# pickles the pieces after each step for debugging. Run "run_pipeline" for the full pipeline
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
