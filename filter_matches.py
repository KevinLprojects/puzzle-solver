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

def ensure_3d(img):
    if img.ndim == 2:
        return img[:, :, None]
    return img

def rotation_angle(v1, v2):
    v1 = np.array(v1, float); v1 /= np.linalg.norm(v1)
    v2 = np.array(v2, float); v2 /= np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1, 1)
    ang = np.arccos(dot)
    if v1[0]*v2[1] - v1[1]*v2[0] < 0:
        ang = -ang
    return ang

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], float)

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

    chA = imgA.shape[2]

    theta = rotation_angle(vA, vB)
    R = rotation_matrix(theta)

    pA = np.array(pA, float)
    pB = np.array(pB, float)

    t = pB - np.matmul(R, pA)

    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]

    cornersA = np.array([[0,0], [wA,0], [wA,hA], [0,hA]], float)
    cornersA_t = np.matmul(cornersA, R.T) + t
    cornersB = np.array([[0,0], [wB,0], [wB,hB], [0,hB]], float)

    all_pts = np.vstack([cornersA_t, cornersB])
    min_x = int(np.floor(all_pts[:,0].min()))
    min_y = int(np.floor(all_pts[:,1].min()))
    max_x = int(np.ceil(all_pts[:,0].max()))
    max_y = int(np.ceil(all_pts[:,1].max()))

    shift = np.array([-min_x, -min_y], float)

    canvas = None
    if img:
        out_w = max_x - min_x
        out_h = max_y - min_y
        
        canvas = np.zeros((out_h, out_w, chA), np.float32)
        weight = np.zeros((out_h, out_w, 1), np.float32)

        MB = np.array([[1,0, shift[0]],
                    [0,1, shift[1]]], float)
        
        MA = np.hstack([R, (t + shift).reshape(2,1)])

        if separate:
            warpedB, maskB = warp_and_mask(imgB, MB, out_w, out_h)
            warpedA, maskA = warp_and_mask(imgA, MA, out_w, out_h)

            if chA == 1:
                warpedB = warpedB[:, :, 0]
                warpedA = warpedA[:, :, 0]

            canvas = (warpedA, warpedB)
        
        else:
            warpedB, maskB = warp_and_mask(imgB, MB, out_w, out_h)
            canvas += warpedB
            weight += maskB

            warpedA, maskA = warp_and_mask(imgA, MA, out_w, out_h)
            canvas += warpedA
            weight += maskA

            canvas = canvas / ((weight == 2) + 1)
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            if chA == 1:
                canvas = canvas[:, :, 0]

    return canvas, R, t, shift

def map_point_from_B(Pb, shift):
    return Pb + shift

def map_point_from_A(Pa, R, t, shift):
    return np.matmul(R, Pa) + t + shift

def project_points_linefit(points):
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    m, b = np.polyfit(x, y, 1)

    direction = np.array([1.0, m])
    direction /= np.linalg.norm(direction)

    centroid = pts.mean(axis=0)
    cx = centroid[0]
    cy = m * cx + b
    line_point = np.array([cx, cy])

    projected = []
    for p in pts:
        v = p - line_point
        t = np.dot(v, direction)
        proj = line_point + t * direction
        projected.append(proj)

    return np.array(projected)


def corner_allignment_score(edgeA, edgeB, R, t, shift, piece_size):
    translated_edgeA = []
    for point in edgeA:
        translated_edgeA.append(map_point_from_A(np.array(point), R, t, shift))

    translated_edgeB = []
    for point in edgeB:
        translated_edgeB.append(map_point_from_B(np.array(point), shift))

    points = project_points_linefit(np.array(translated_edgeA + translated_edgeB))
    translated_edgeA = [points[0], points[1]]
    translated_edgeB = [points[2], points[3]]

    max_dist = 0
    for pointB in translated_edgeB:
        min_dist = float('inf')
        for pointA in translated_edgeA:
            dist = np.linalg.norm(pointB - pointA)
            if dist < min_dist:
                min_dist = dist
        if min_dist > max_dist:
            max_dist = min_dist
    
    return max_dist / np.sqrt(piece_size)

def slope_to_angle_diff(slope1, slope2):
    a1 = np.degrees(np.arctan2(slope1, 1)) % 360
    a2 = np.degrees(np.arctan2(slope2, 1)) % 360

    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)

def piece_allignment_angle_diff(polyA, polyB, R, t, shift):
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

    return slope_to_angle_diff(mA, mB)

def rough_check(socket_piece, socket, plug_piece, plug, piece_size):
    img1 = cv.polylines(plug_piece.image.copy(), [np.array(plug[2]).reshape((-1, 1, 2))], False, (255, 0, 0), 2)
    img2 = cv.polylines(socket_piece.image.copy(), [np.array(socket[2]).reshape((-1, 1, 2))], False, (0, 255, 0), 2)

    img, R, t, shift = align_and_combine(img1, plug[0], plug[1], img2, socket[0], socket[1])
    edge_score = corner_allignment_score(plug[2], socket[2], R, t, shift, piece_size)
    angle_score = piece_allignment_angle_diff(plug_piece.poly, socket_piece.poly, R, t, shift)

    if edge_score < EDGE_ALLIGNMENT_THRESHOLD and angle_score < PIECE_ALLIGNMENT_ANGLE_THRESHOLD:
        return True
    else:
        return False

def display_match(match):
    img2, R, t, shift = align_and_combine(match.plug_piece.image.copy(), match.plug[0], match.plug[1], match.socket_piece.image.copy(), match.socket[0], match.socket[1], img=True)

    return img2

def generate_matches(piece_list, piece_size):
    good = 0
    total = 0
    for i_socket_piece, socket_piece in tqdm(enumerate(piece_list)):
        for socket in socket_piece.sockets:
            socket.append([])
            for i_plug_piece, plug_piece in enumerate(piece_list):
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