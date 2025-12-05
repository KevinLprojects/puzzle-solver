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
EDGE_BLUR_PERCENT = 0.1
DERIVATIVE_PERCENT = 0.15

def odd_size(size):
    return int(2 * np.floor(size / 2 + 1) - 1)

def radial_derivative(contour, center, idx, n=3):
    contour = contour[:,0,:]
    cx, cy = center
    N = len(contour)

    radii = np.sqrt((contour[:,0]-cx)**2 + (contour[:,1]-cy)**2)

    contributions = []
    weights = []

    for k in range(1, n+1):
        w = 1 - (k / n)

        i1 = (idx - k) % N
        i2 = (idx - (k+1)) % N

        p1 = contour[i1]
        p2 = contour[i2]
        ds_left = np.linalg.norm(p1 - p2)
        if ds_left > 0:
            dr_left = radii[i1] - radii[i2]
            contributions.append(dr_left / ds_left)
            weights.append(w)

        j1 = (idx + k) % N
        j2 = (idx + (k+1)) % N

        p1 = contour[j1]
        p2 = contour[j2]
        ds_right = np.linalg.norm(p1 - p2)
        if ds_right > 0:
            dr_right = radii[j2] - radii[j1]
            contributions.append(dr_right / ds_right)
            weights.append(w)

    if not contributions:
        return 0.0

    contributions = np.array(contributions)
    weights = np.array(weights)
    return float(np.sum(contributions * weights) / np.sum(weights))

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

def combos_of_four(zero_crossings):
    if len(zero_crossings) < 4:
        return None
    
    return list(combinations(zero_crossings, 4))

def segments_intersect(A, B, C, D):
    def ccw(P, Q, R):
        return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def polygon_self_intersects(poly):
    P0, P1, P2, P3 = poly

    edges = [
        (P0, P1),
        (P1, P2),
        (P2, P3),
        (P3, P0)
    ]

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

def first_valid_quad(contour, combo):
    poly = np.array([contour[i][0] for i in combo], dtype=np.int32)

    if polygon_self_intersects(poly):
        return None

    return poly

def cost(cleaned_mask, poly):
    try:
        filled_poly = cv.fillPoly(np.zeros_like(cleaned_mask), [poly], 1)
    except:
        return None

    return np.sum(np.logical_xor(cleaned_mask, filled_poly))

def contour_polygon(piece, piece_size):
    mask = piece.mask
    contour = piece.contour
    M = cv.moments(piece.contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    N = len(contour)
    derivatives = np.array([radial_derivative(contour, (cx, cy), i, n=int(DERIVATIVE_PERCENT * np.sqrt(piece_size))) for i in range(N)])

    zero_ids = find_pos_to_neg_zero_crossings(derivatives)

    combos = combos_of_four(zero_ids)

    if combos is None:
        return None
    
    size = odd_size(np.sqrt(piece_size) * KERNEL_SIZE_PERCENT)
    kernel = np.ones((size,size),np.uint8)
    closed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    opened_mask = cv.morphologyEx(closed_mask, cv.MORPH_OPEN, kernel)
    
    best_cost = float('inf')
    best_poly = None
    
    for combo in combos:
        poly = first_valid_quad(contour, combo)
        current_cost = cost(opened_mask, poly)

        if current_cost is not None and current_cost < best_cost:
            best_cost = current_cost
            best_poly = poly

    return best_poly

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

def closest_edge_normal(point, polygon):
    poly = np.squeeze(polygon)
    x0, y0 = point

    min_dist = float('inf')
    best_normal = None
    best_edge = None
    best_proj = None

    N = len(poly)

    for i in range(N):
        p1 = poly[i]
        p2 = poly[(i+1) % N]

        v = p2 - p1
        w = np.array([x0, y0]) - p1
        L = np.dot(v, v)

        t = np.clip(np.dot(w, v) / L, 0.0, 1.0)
        proj = p1 + t * v

        dist = np.linalg.norm(proj - np.array([x0, y0]))
        if dist < min_dist:
            min_dist = dist
            best_proj = proj
            
            dx, dy = v
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

    dot1 = np.dot(vec_to_point, n1)
    dot2 = np.dot(vec_to_point, n2)

    chosen = n1 if dot1 > dot2 else n2

    return chosen, best_edge


def add_sockets_and_plugs(piece, piece_size):
    mask = piece.mask
    poly = piece.poly

    filled_poly = np.zeros_like(mask)
    cv.fillPoly(filled_poly, [poly], 1)

    diff = mask.astype(np.int8) - filled_poly.astype(np.int8)
    plugs = diff > 0
    sockets = diff < 0

    plug_points = feature_points(plugs, piece_size)
    socket_points = feature_points(sockets, piece_size)

    piece.plugs = []
    piece.sockets = []

    for plug in plug_points:
        normal, edge = closest_edge_normal(plug, piece.poly)
        piece.plugs.append([list(plug), list(normal), list(edge)])

    for socket in socket_points:
        normal, edge = closest_edge_normal(socket, piece.poly)
        piece.sockets.append([list(socket), list(normal), list(edge)])
    
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