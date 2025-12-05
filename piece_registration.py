import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model
import cv2 as cv

LOCAL_MODEL_PATH = "./sam3_checkpoint"
IMAGE_PATH = "100_piece_puzzle.jpg"
TEXT_PROMPT = "puzzle piece"

SCORE_THRESHOLD = 0.9 # SAM confidence threshold
OVERLAP_PERCENT = 0.5 # overlap of 1024x1024 SAM tiles

PIECE_SIZE_THRESHOLD = 0.5 # percentage of the median piece size to be accepted as a piece

PAD_PERCENT = 0.25 # percentage of smallest dimension to add as padding

class Match:
    def __init__(self, socket_piece, socket, plug_piece, plug):
        self.socket_piece = socket_piece
        self.socket = socket
        self.plug_piece = plug_piece
        self.plug = plug
        self.overlap_score = None
        self.underlap_score = None
        self.color_score = None
        self.total_score = None
    
    def get_transform_info(self):
        pA = self.plug[0]
        vA = self.plug[1]

        pB = self.socket[0]
        vB = self.socket[1]

        return (self.plug_piece, pA, vA, self.socket_piece, pB, vB)

class Piece:
    def __init__(self, x, y, w, h, mask):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mask = mask
        self.image = None
        self.image_extended = None
        self.contour = None
        self.poly = None
        self.plugs = []
        self.sockets = []
    
    def pad_mask(self):
        if self.w > self.h:
            pad = int(np.round(self.h * PAD_PERCENT))
        else:
            pad = int(np.round(self.w * PAD_PERCENT))

        self.w += 2*pad
        self.h += 2*pad

        self.x -= pad
        self.y -= pad

        self.mask = np.pad(self.mask, ((pad, pad), (pad, pad)), mode='constant', constant_values=0).astype(np.uint8)
    
    def clean_mask(self):
        self.mask = cv.GaussianBlur(self.mask * 255, (13, 13), 0)
        self.mask = (self.mask > 128).astype(np.uint8)
    
    def get_cleaned_contour(self):
        contours, _ = cv.findContours(self.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        self.contour = max(contours, key=cv.contourArea)
    
    def outpaint(self):
        image_extended = cv.inpaint(cv.medianBlur(self.image, 7), 1-self.mask, 5, cv.INPAINT_TELEA)
        image_extended = cv.medianBlur(image_extended, 5)
        self.image_extended = image_extended * np.stack((1-self.mask, 1-self.mask, 1-self.mask), axis=2) + self.image
    
    def display(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(self.mask, cmap='gray')
        ax1.axis('off')
        ax2.imshow(self.image)
        ax2.axis('off')
        ax3.imshow(self.image_extended)
        ax3.axis('off')
        plt.show()

class Slice:
    def __init__(self, content, x, y, w, h):
        self.content = content
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def display(self):
        plt.imshow(self.content)
        plt.show()

def get_slices(image, w, h, overlap_percentage=OVERLAP_PERCENT):
    slice_list = []
    tile_size = 1024
    step = tile_size - int(tile_size * overlap_percentage)

    y = 0
    while y < h:
        x = 0
        while x < w:
            tile = image[y:y+tile_size, x:x+tile_size]

            padded = np.pad(tile, ((0, tile_size - tile.shape[0]), (0, tile_size - tile.shape[1]), (0, 0)), mode='constant')

            slice_list.append(Slice(padded, x, y, tile_size, tile_size))
            x += step

        y += step

    return slice_list

def inference(image, image_width, image_height, model, processor, device):
    inputs = processor(
        images=image, 
        text=TEXT_PROMPT, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Converts raw output tensors into masks/boxes scaled to original image size
    results = processor.post_process_instance_segmentation(
        outputs, 
        threshold=SCORE_THRESHOLD, 
        target_sizes=[(image_height, image_width)]
    )[0]

    masks = results["masks"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    if len(masks) == 0:
        return None, None

    masks_filtered = []

    for mask, score in zip(masks, scores):
        if score < SCORE_THRESHOLD:
            continue
        else:
            masks_filtered.append(mask)

    median_size = np.median(np.sum(np.array(masks_filtered), axis=(1,2)))
    
    mask = np.sum(np.array(masks_filtered).astype(np.uint8), axis=0)
    mask = mask > 0

    return mask, median_size

def get_piece_mask(image, width, height):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SAM 3 model from {LOCAL_MODEL_PATH}...")
    model = Sam3Model.from_pretrained(LOCAL_MODEL_PATH).to(device)
    processor = Sam3Processor.from_pretrained(LOCAL_MODEL_PATH)
    
    slice_list = get_slices(image, width, height)

    output_mask = np.zeros(shape=(image.shape[0], image.shape[1])).astype(np.uint8)

    median_list = []

    for tile in tqdm(slice_list, desc="segmenting tiles"):
        mask, median_size = inference(tile.content, tile.w, tile.h, model, processor, device)
        if mask is None:
            continue

        median_list.append(median_size)
        max_x = tile.x + 1024
        max_y = tile.y + 1024
        if max_x > width:
            max_x = width
        if max_y > height:
            max_y = height
        
        output_mask[tile.y:max_y, tile.x:max_x] += mask[0:max_y-tile.y, 0:max_x-tile.x]
    
    piece_size = np.median(np.array(median_list))

    if piece_size < 10000:
        raise ValueError("not enough piece resolution")

    return (output_mask > 0).astype(np.uint8), piece_size

def sepparate_contours(image, height, width, mask, piece_size):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    piece_list = []

    for contour in contours:
        piece_mask = np.zeros((height, width), dtype=np.uint8)
        cv.drawContours(piece_mask, [contour], contourIdx=-1, color=255, thickness=-1)

        x, y, w, h = cv.boundingRect(contour)

        cropped_mask = mask[y:y+h, x:x+w]

        size = np.sum(cropped_mask > 0)

        if size > piece_size + PIECE_SIZE_THRESHOLD * piece_size or size < piece_size - PIECE_SIZE_THRESHOLD * piece_size:
            continue

        new_piece = Piece(x, y, w, h, cropped_mask)

        new_piece.pad_mask()
        new_piece.clean_mask()
        new_piece.image = image[new_piece.y:new_piece.y+new_piece.h, new_piece.x:new_piece.x+new_piece.w] * np.stack((new_piece.mask, new_piece.mask, new_piece.mask), axis=2)
        new_piece.get_cleaned_contour()
        new_piece.outpaint()

        piece_list.append(new_piece)
    
    return piece_list

if __name__ == "__main__":
    image = Image.open(IMAGE_PATH).convert("RGB")
    width, height = image.size
    image = np.array(image)
    print(f"Loaded image: {IMAGE_PATH} ({width}x{height})")

    mask, piece_size = get_piece_mask(image, width, height)
    with open('puzzle_mask.pkl', 'wb') as f:
        pickle.dump((mask, piece_size), f)

    with open('puzzle_mask.pkl', 'rb') as f:
        mask, piece_size = pickle.load(f)

    piece_list = sepparate_contours(image, height, width, mask, piece_size)

    print("number of pieces: ", len(piece_list))

    with open('puzzle_pieces.pkl', 'wb') as f:
        pickle.dump(piece_list, f)