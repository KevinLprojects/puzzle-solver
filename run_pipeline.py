from piece_registration import get_piece_mask, sepparate_contours
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

image = Image.open('100_piece_puzzle.jpg').convert("RGB")
width, height = image.size
image = np.array(image)

mask, piece_size = get_piece_mask(image, width, height)

piece_list = sepparate_contours(image, height, width, mask, piece_size)
print("number of pieces: ", len(piece_list))

if DISPLAY:
    piece_list[0].display()

for piece in tqdm(piece_list, desc="fitting polygons"):
    piece.poly = contour_polygon(piece, piece_size)

for piece in piece_list:
    add_sockets_and_plugs(piece, piece_size)

if DISPLAY:
    display_features(piece_list[0])

num_matches = generate_matches(piece_list, piece_size)

pbar = tqdm(total=num_matches, desc="sorting matches")
socket_list = []
for piece in piece_list:
    for socket in piece.sockets:
        socket_list.append(socket)
        socket[-1].sort(key=partial(cost, piece_size=piece_size, pbar=pbar))

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