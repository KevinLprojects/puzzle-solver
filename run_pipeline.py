"""
Course Number: ENGR 13300
Semester: e.g. Spring 2025

Description:
    Runs the full pipeline from taking in input image to displaying sorted output matches.

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

from piece_registration import get_piece_mask, separate_contours
from feature_detection import contour_polygon, add_sockets_and_plugs, display_features
from filter_matches import generate_matches, display_match
from rank_matches import cost, sort_sockets

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from functools import partial

DISPLAY = True

# get input image
image = Image.open('100_piece_puzzle.jpg').convert("RGB")
width, height = image.size
image = np.array(image)

# get the puzzle piece mask and median piece size
mask, piece_size = get_piece_mask(image, width, height)

# find and separate all the contours in the mask
piece_list = separate_contours(image, height, width, mask, piece_size)
print("number of pieces: ", len(piece_list))

if DISPLAY:
    # display the first piece's mask, image, and outpainted image
    piece_list[0].display()

# fit polygons to each piece
for piece in tqdm(piece_list, desc="fitting polygons"):
    piece.poly = contour_polygon(piece, piece_size)

# find the plugs and sockets for each piece
for piece in piece_list:
    add_sockets_and_plugs(piece, piece_size)

if DISPLAY:
    # display the first piece's polygon and plugs/sockets
    display_features(piece_list[0])

# filter matches
num_matches = generate_matches(piece_list, piece_size)

# sort every sockets matches by their score
pbar = tqdm(total=num_matches, desc="sorting matches")
socket_list = []
for piece in piece_list:
    for socket in piece.sockets:
        socket_list.append(socket)
        # using partial to pass in the piece size and pbar to the cost function
        socket[-1].sort(key=partial(cost, piece_size=piece_size, pbar=pbar))

# sort the sockets by their best matches score
socket_list.sort(key=sort_sockets)

# for each match display an arrow showing the connection and the composite image between the two pieces
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
