from piece_registration import Match

import numpy as np
import cv2 as cv
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import sys
sys.setrecursionlimit(3000)

EDGE_ALLIGNMENT_THRESHOLD = 0.15 # decimal percent
PIECE_ALLIGNMENT_ANGLE_THRESHOLD = 10 # degrees

# adds an extra dimension to grayscale images for convenience
def ensure_3d(img):
    if img.ndim == 2:
        return img[:, :, None]
    return img

# get the angle between two vectos
def rotation_angle(v1, v2):
    # normalize the vectors to remove magnitude component of dot product
    v1 = np.array(v1, float)
    v1 /= np.linalg.norm(v1)

    v2 = np.array(v2, float)
    v2 /= np.linalg.norm(v2)

    # clip the dot product just incase
    dot = np.clip(np.dot(v1, v2), -1, 1)
    ang = np.arccos(dot)

    # make sure the angle is counter clockwise (cross product)
    if v1[0]*v2[1] - v1[1]*v2[0] < 0:
        ang = -ang
    return ang

# create a rotation matrix from rotation angle
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], float)

# warps image according to matrix and get transformed mask
def warp_and_mask(img, M, out_w, out_h):
    warped = cv.warpAffine(img, M, (out_w, out_h),
                            flags=cv.INTER_LINEAR,
                            borderValue=0)

    warped = ensure_3d(warped)

    mask = (np.any(warped > 0, axis=2, keepdims=True)).astype(np.float32)

    return warped.astype(np.float32), mask

# transforms imageA onto imageB using a point and vector to align on both
def align_and_combine(imgA, pA, vA, imgB, pB, vB, img=False, separate=False):
    imgA = ensure_3d(imgA)
    imgB = ensure_3d(imgB)

    # get the number of channels
    chA = imgA.shape[2]

    theta = rotation_angle(vA, vB)
    R = rotation_matrix(theta)

    pA = np.array(pA, float)
    pB = np.array(pB, float)

    # linear tranformation of A
    t = pB - np.matmul(R, pA)

    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]

    # get the final coords of the corners of both pieces bounding boxes in B's coord space
    cornersA = np.array([[0,0], [wA,0], [wA,hA], [0,hA]], float)
    cornersA_t = np.matmul(cornersA, R.T) + t
    cornersB = np.array([[0,0], [wB,0], [wB,hB], [0,hB]], float)

    # get the bounding box that fits both rotated bounding boxes
    # I do this regardless of img=True/False, because the coords and transform is relative to this bounding box
    all_pts = np.vstack([cornersA_t, cornersB])
    min_x = int(np.floor(all_pts[:,0].min()))
    min_y = int(np.floor(all_pts[:,1].min()))
    max_x = int(np.ceil(all_pts[:,0].max()))
    max_y = int(np.ceil(all_pts[:,1].max()))

    # shift to apply to B to coord space of the larger bounding box
    shift = np.array([-min_x, -min_y], float)

    # outputing an image
    canvas = None
    if img:
        out_w = max_x - min_x
        out_h = max_y - min_y
        
        canvas = np.zeros((out_h, out_w, chA), np.float32)

        # where the masks overlap I have average pixel values
        weight = np.zeros((out_h, out_w, 1), np.float32)

        # transformation matrix to apply to B (no rotation)
        MB = np.array([[1,0, shift[0]],
                    [0,1, shift[1]]], float)
        
        # transformation matrix to apply to A 
        # t = shift relative to B
        # shift = shift of B relative to new bounding box
        # R = rotation matrix of A to allign with B's feature vector
        MA = np.hstack([R, (t + shift).reshape(2,1)])
        
        # return separate images for A and B (if I need to do processing to them separately before combining)
        if separate:
            warpedB, maskB = warp_and_mask(imgB, MB, out_w, out_h)
            warpedA, maskA = warp_and_mask(imgA, MA, out_w, out_h)

            if chA == 1:
                warpedB = warpedB[:, :, 0]
                warpedA = warpedA[:, :, 0]

            canvas = (warpedA, warpedB)
        
        # if a single image is needed, I sum the images and masks and devide by two in the areas with overlap
        else:
            warpedB, maskB = warp_and_mask(imgB, MB, out_w, out_h)
            canvas += warpedB
            weight += maskB

            warpedA, maskA = warp_and_mask(imgA, MA, out_w, out_h)
            canvas += warpedA
            weight += maskA
            
            # if the input was grayscale
            if chA == 1:
                # get rid of extra axis
                canvas = canvas[:, :, 0]
                canvas = canvas > 0
            
            else:
                canvas = canvas / ((weight == 2) + 1)
                canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return canvas, R, t, shift

# transform point from B's mask onto combined mask (same logic as above)
def map_point_from_B(Pb, shift):
    return Pb + shift

# transform point from A's mask onto combined mask (same logic as above)
def map_point_from_A(Pa, R, t, shift):
    return np.matmul(R, Pa) + t + shift

# project points onto a linefit of them
# this is used to see how miss aligned two edges are
def project_points_linefit(points):
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    # fit line
    m, b = np.polyfit(x, y, 1)

    # get normalized vector in direction of line fit
    direction = np.array([1.0, m])
    direction /= np.linalg.norm(direction)

    # get the center of the line 
    centroid = pts.mean(axis=0)
    cx = centroid[0]
    cy = m * cx + b
    line_point = np.array([cx, cy])

    projected = []
    for p in pts:
        # remove offset to use standard vec proj formula
        v = p - line_point
        t = np.dot(v, direction)
        # standard vector projection formula
        proj = line_point + t * direction
        projected.append(proj)

    return np.array(projected)

# score for how well corners allign
def corner_allignment_score(edgeA, edgeB, R, t, shift, piece_size):
    # translate the poly points for each pice onto their shared bounding box
    translated_edgeA = []
    for point in edgeA:
        translated_edgeA.append(map_point_from_A(np.array(point), R, t, shift))

    translated_edgeB = []
    for point in edgeB:
        translated_edgeB.append(map_point_from_B(np.array(point), shift))

    # get the projected points
    points = project_points_linefit(np.array(translated_edgeA + translated_edgeB))
    translated_edgeA = [points[0], points[1]]
    translated_edgeB = [points[2], points[3]]

    # get the largest distance between two corner points
    max_dist = 0
    for pointB in translated_edgeB:
        # the min dist here is the dist between the two cooresponding corner points
        min_dist = float('inf')
        for pointA in translated_edgeA:
            dist = np.linalg.norm(pointB - pointA)
            if dist < min_dist:
                min_dist = dist
        if min_dist > max_dist:
            max_dist = min_dist
    
    return max_dist / np.sqrt(piece_size)

# converts two slopes into a difference between their angles
def slope_to_angle_diff(slope1, slope2):
    # get angles of the vectors
    a1 = np.degrees(np.arctan2(slope1, 1)) % 360
    a2 = np.degrees(np.arctan2(slope2, 1)) % 360

    # get the angle diff less than 180
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)

# check how well two pieces allign, how rectalinear they are
def piece_allignment_angle_diff(polyA, polyB, R, t, shift):
    # get the translated poly points
    translated_polyA = []
    for point in polyA:
        translated_polyA.append(map_point_from_A(np.array(point), R, t, shift))

    translated_polyB = []
    for point in polyB:
        translated_polyB.append(map_point_from_B(np.array(point), shift))

    translated_polyA = np.array(translated_polyA)
    translated_polyB = np.array(translated_polyB)

    x = translated_polyA[:, 0]
    y = translated_polyA[:, 1]

    mA, b = np.polyfit(x, y, 1)

    x = translated_polyB[:, 0]
    y = translated_polyB[:, 1]

    mB, b = np.polyfit(x, y, 1)

    # angle diff between the center line of the two pieces
    return slope_to_angle_diff(mA, mB)

# create match for plug socket combo that passes the rough check
def rough_check(socket_piece, socket, plug_piece, plug, piece_size):
    img1 = cv.polylines(plug_piece.image.copy(), [np.array(plug[2]).reshape((-1, 1, 2))], False, (255, 0, 0), 2)
    img2 = cv.polylines(socket_piece.image.copy(), [np.array(socket[2]).reshape((-1, 1, 2))], False, (0, 255, 0), 2)

    img, R, t, shift = align_and_combine(img1, plug[0], plug[1], img2, socket[0], socket[1])
    edge_score = corner_allignment_score(plug[2], socket[2], R, t, shift, piece_size)
    angle_score = piece_allignment_angle_diff(plug_piece.poly, socket_piece.poly, R, t, shift)

    # thresholds for how well the edges and angles line up
    if edge_score < EDGE_ALLIGNMENT_THRESHOLD and angle_score < PIECE_ALLIGNMENT_ANGLE_THRESHOLD:
        return True
    else:
        return False

# displays the piece images composited together
def display_match(match):
    img2, R, t, shift = align_and_combine(match.plug_piece.image.copy(), match.plug[0], match.plug[1], match.socket_piece.image.copy(), match.socket[0], match.socket[1], img=True)

    return img2

# create matches for all socket and plug combos that pass the rough check
def generate_matches(piece_list, piece_size):
    good = 0
    total = 0
    for i_socket_piece, socket_piece in tqdm(enumerate(piece_list)):
        for socket in socket_piece.sockets:
            socket.append([])
            for i_plug_piece, plug_piece in enumerate(piece_list):
                # don't match piece to itself
                if i_plug_piece == i_socket_piece:
                    continue
                for plug in plug_piece.plugs:
                    total += 1
                    if rough_check(socket_piece, socket, plug_piece, plug, piece_size):
                        good += 1
                        match = Match(socket_piece, socket, plug_piece, plug)
                        # display_match(match)
                        socket[-1].append(match)
    
    print("total matches: ", total, " good matches: ", good)
    return good

# pickles the pieces after each step for debugging. Run "run_pipeline" for the full pipeline
if __name__ == "__main__":
    with open('puzzle_mask.pkl', 'rb') as f:
        mask, piece_size = pickle.load(f)

    with open('puzzle_pieces.pkl', 'rb') as f:
        piece_list = pickle.load(f)

    num_matches = generate_matches(piece_list, piece_size)

    with open('puzzle_pieces.pkl', 'wb') as f:
        pickle.dump(piece_list, f)

    with open('num_matches.pkl', 'wb') as f:
        pickle.dump(num_matches, f)