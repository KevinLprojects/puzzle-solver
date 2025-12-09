"""
Course Number: ENGR 13300
Semester: e.g. Spring 2025

Description:
    Finds the corners of the puzzle pieces, fits a polygon to the corners, and finds the centers and edge normals of the features

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

from piece_registration import Piece

import numpy as np
import cv2 as cv
import pickle
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

KERNEL_SIZE_PERCENT = 0.5 # percent of the sqrt of the piece size for the kernel size
KERNEL_SIZE_PERCENT_FEATURES = 0.15 # percent of the sqrt of the piece size for the kernel size for feature erosion
DERIVATIVE_PERCENT = 0.15 # percent of sqrt of piece size points to include in a single derivative

# returns the closest odd int to the given input
def odd_size(size):
    return int(2 * np.floor(size / 2 + 1) - 1)

# calculates dr/dtheta for a given point in s set of coords and a center point. n is the number of neighboring pixels to use for a given pixels derivative
def radial_derivative(contour, center, idx, n=3):
    # opencv contours have shape [N, 1, 2]
    contour = contour[:,0,:]
    cx, cy = center
    N = len(contour)

    # find the distance of each coord in the contour from the center
    radii = np.sqrt((contour[:,0]-cx)**2 + (contour[:,1]-cy)**2)

    contributions = []
    weights = []

    # loops through the neighboring pixels
    for k in range(1, n+1):
        # weight according to distance from given id
        w = 1 - (k / n)

        # get two consecutive neighbors from the left side
        i1 = (idx - k) % N
        i2 = (idx - (k+1)) % N

        # get the coords of those points
        p1 = contour[i1]
        p2 = contour[i2]

        # distance between the two points
        ds_left = np.linalg.norm(p1 - p2)

        # make sure they are 2 different points, then record dr/ds and the weight
        if ds_left > 0:
            dr_left = radii[i1] - radii[i2]
            contributions.append(dr_left / ds_left)
            weights.append(w)

        # do the same thing on the right side
        j1 = (idx + k) % N
        j2 = (idx + (k+1)) % N

        p1 = contour[j1]
        p2 = contour[j2]
        ds_right = np.linalg.norm(p1 - p2)
        if ds_right > 0:
            dr_right = radii[j2] - radii[j1]
            contributions.append(dr_right / ds_right)
            weights.append(w)

    contributions = np.array(contributions)
    weights = np.array(weights)
    # return the weighted average of the derivative
    return float(np.sum(contributions * weights) / np.sum(weights))

# find the points where a sequence crosses 0 (only from positive to negative or the convex corners)
def find_pos_to_neg_zero_crossings(values):
    N = len(values)
    crossings = []

    for i in range(N):
        a = values[i]
        b = values[(i + 1) % N]

        if a > 0 and b < 0:
            crossings.append(i)

        elif a == 0 and b < 0:
            crossings.append(i)

    return crossings

# get all the combos of 4 points from a given set of points for fitting a quad
def combos_of_four(zero_crossings):
    if len(zero_crossings) < 4:
        return None
    
    return list(combinations(zero_crossings, 4))

# check if two line segments intersect (I found this on the internet don't sue me)
def segments_intersect(A, B, C, D):
    def ccw(P, Q, R):
        return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

# check if a polygon self intersects (sensitive to order)
def polygon_self_intersects(poly):
    P0, P1, P2, P3 = poly

    edges = [
        (P0, P1),
        (P1, P2),
        (P2, P3),
        (P3, P0)
    ]

    # combos of edges to check (adjacent edges can't intersect for obvious reasons)
    checks = [
        (0, 2),
        (1, 3),
    ]

    for a, b in checks:
        A1, A2 = edges[a]
        B1, B2 = edges[b]
        if segments_intersect(A1, A2, B1, B2):
            return True
    
    return False

# check if a combo is a valid quad
def valid_quad(contour, combo):
    poly = np.array([contour[i][0] for i in combo], dtype=np.int32)

    if polygon_self_intersects(poly):
        return None

    return poly

def cost(cleaned_mask, poly):
    # create a filled polygon to compare with the piece
    try:
        filled_poly = cv.fillPoly(np.zeros_like(cleaned_mask), [poly], 1)

    # if something weird happens
    except:
        return None

    # return the non matching area
    return np.sum(np.logical_xor(cleaned_mask, filled_poly))

def contour_polygon(piece, piece_size):
    mask = piece.mask
    contour = piece.contour

    # get the center of mass of the contour
    M = cv.moments(piece.contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    N = len(contour)
    derivatives = np.array([radial_derivative(contour, (cx, cy), i, n=int(DERIVATIVE_PERCENT * np.sqrt(piece_size))) for i in range(N)])

    zero_ids = find_pos_to_neg_zero_crossings(derivatives)

    # create combos out of the convex max points on the contour
    combos = combos_of_four(zero_ids)

    if combos is None:
        return None
    
    # close then open the mask to get rid of the plugs and sockets
    size = odd_size(np.sqrt(piece_size) * KERNEL_SIZE_PERCENT)
    kernel = np.ones((size,size),np.uint8)
    closed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    opened_mask = cv.morphologyEx(closed_mask, cv.MORPH_OPEN, kernel)
    
    best_cost = float('inf')
    best_poly = None
    
    # loops through each combo to find the one with the lowest cost
    for combo in combos:
        poly = valid_quad(contour, combo)
        current_cost = cost(opened_mask, poly)

        if current_cost is not None and current_cost < best_cost:
            best_cost = current_cost
            best_poly = poly

    return best_poly

# for a given mask, find the centers of the individual contours (after eroding to separate them)
def feature_points(mask, piece_size):
    size = odd_size(np.sqrt(piece_size) * KERNEL_SIZE_PERCENT_FEATURES)
    kernel = np.ones((size,size), np.uint8)
    erosion = cv.erode(mask.astype(np.uint8), kernel, iterations = 1)

    contours, _ = cv.findContours(erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    points = []
    for contour in contours:
        try:
            M = cv.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            points.append((cx, cy))
        except:
            print("make the feature kernel bigger, something weird is happening")
            sys.exit()
    
    return points

# find the normal of the closest edge to a point
def closest_edge_normal(point, polygon):
    # squeeze to git rid of the weird extra dimension opencv adds
    poly = np.squeeze(polygon)
    x0, y0 = point

    min_dist = float('inf')
    best_normal = None
    best_edge = None
    best_proj = None

    N = len(poly)

    # go through each pair of consecutive points to find the closest ones
    for i in range(N):
        p1 = poly[i]
        # if it raps around, get that point
        p2 = poly[(i+1) % N]


        v = p2 - p1
        w = np.array([x0, y0]) - p1
        L = np.dot(v, v)

        # project the input point onto the edge 
        t = np.clip(np.dot(w, v) / L, 0.0, 1.0)
        proj = p1 + t * v

        # distance between input point and projection (it might be fine to just get the closest two points but whatever)
        dist = np.linalg.norm(proj - np.array([x0, y0]))
        if dist < min_dist:
            min_dist = dist
            best_proj = proj
            
            dx, dy = v
            # get both normals (the polygon can be either clockwise or counterclockwise)
            n1 = np.array([-dy, dx])
            n2 = np.array([dy, -dx])

            if np.linalg.norm(n1) > 0:
                n1 = n1 / np.linalg.norm(n1)
                n2 = n2 / np.linalg.norm(n2)

            best_normal = (n1, n2)
            best_edge = [p1, p2]

    n1, n2 = best_normal
    proj = best_proj
    
    
    vec_to_point = np.array([x0, y0]) - proj

    # ge the normal that points closest in direction to the input point (sockets point in, plugs point out)
    dot1 = np.dot(vec_to_point, n1)
    dot2 = np.dot(vec_to_point, n2)

    chosen = n1 if dot1 > dot2 else n2

    return chosen, best_edge

# find the plugs and sockets and add them to the piece objects
def add_sockets_and_plugs(piece, piece_size):
    mask = piece.mask
    poly = piece.poly

    filled_poly = np.zeros_like(mask)
    cv.fillPoly(filled_poly, [poly], 1)

    # in the diff between the polygon and the mask, the plugs will be positive and the sockets will be negative
    diff = mask.astype(np.int8) - filled_poly.astype(np.int8)
    plugs = diff > 0
    sockets = diff < 0

    # get the points for each plug and socket
    plug_points = feature_points(plugs, piece_size)
    socket_points = feature_points(sockets, piece_size)

    piece.plugs = []
    piece.sockets = []

    # add the plugs sockets and their normals
    for plug in plug_points:
        normal, edge = closest_edge_normal(plug, piece.poly)
        piece.plugs.append([list(plug), list(normal), list(edge)])

    for socket in socket_points:
        normal, edge = closest_edge_normal(socket, piece.poly)
        piece.sockets.append([list(socket), list(normal), list(edge)])

# displays the polygon and feature points on the piece image
def display_features(piece):
    img = piece.image
    cv.polylines(img, [piece.poly], True, (0, 255, 0), 2)

    for socket in piece.sockets:
        cv.circle(img, socket[0], 3, (0, 0, 255), -1)
        cv.arrowedLine(img, socket[0], (socket[0][0] + int(30 * socket[1][0]), socket[0][1] + int(30 * socket[1][1])), (0, 0, 255), 2, cv.LINE_AA, 0, 0.1)
    for plug in piece.plugs:
        cv.circle(img, plug[0], 3, (255, 0, 0), -1)
        cv.arrowedLine(img, plug[0], (plug[0][0] + int(30 * plug[1][0]), plug[0][1] + int(30 * plug[1][1])), (255, 0, 0), 2, cv.LINE_AA, 0, 0.1)
    
    plt.imshow(img)
    plt.show()

# pickles the pieces after each step for debugging. Run "run_pipeline" for the full pipeline
if __name__ == "__main__":
    with open('puzzle_mask.pkl', 'rb') as f:
        mask, piece_size = pickle.load(f)

    with open('puzzle_pieces.pkl', 'rb') as f:
        piece_list = pickle.load(f)

    for piece in tqdm(piece_list, desc="fitting polygons"):
        piece.poly = contour_polygon(piece, piece_size)

    for piece in piece_list:
        add_sockets_and_plugs(piece, piece_size)

    with open('puzzle_pieces.pkl', 'wb') as f:
        pickle.dump(piece_list, f)
