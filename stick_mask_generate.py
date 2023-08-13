import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import glob
from pathlib import Path

import os
from segment_anything import sam_model_registry, SamPredictor
from mask_to_bbox import mask_to_bbox

folder_name = 'temp_NHL_dataset'

output_dir = folder_name.replace('temp', 'new')
#print(output_dir)

if not os.path.exists(output_dir):
   # Create a new directory because it does not exist
   os.makedirs(output_dir)
   print("The new directory 1 is created!")

input_images_path = os.path.join(folder_name, 'images/')
print(input_images_path)

image_path_list = glob.glob(input_images_path + "*")
image_path_list.sort()

# player bounding box mask
input_player_bbox_path = os.path.join(folder_name, 'player_only_mask/')

player_bbox_path_list = glob.glob(input_player_bbox_path + "*")
player_bbox_path_list.sort()


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
input_label = np.array([1])

debug = False

global close
global clone, image
global mask, mask_color

global result

global option

global draw_point_coord, erase_point_coord

global cache

global player_bbox

# ininit 
close = False
clone, image = None, None
mask, mask_color = None, None
result = None
option = 'View'

draw_point_coord = []
erase_point_coord = []

cache = None

def get_mask(mask):

    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * 255
    mask_image = mask_image.astype(np.uint8)

    return mask_image


def create_color_hightlight(image, color):

    cyan = np.full_like(image, color)

    blend = 0.5
    color_image = cv2.addWeighted(image, blend, cyan, 1-blend, 0)

    return color_image

def apply_mask_to_image(image, mask, mask_color):

    return np.where(mask>=240, mask_color, image)


def add_mask(mask1, mask2):
    
    result = mask1 + mask2
    result[result < mask2] = 255

    return result

def determine_thickness(image):

    h, w, _ = image.shape
    if w <= 250:
        return 2
    elif w > 250 and w <= 400:
        return 3
    elif w > 400 and w <= 550:
        return 4
    elif w > 550 and w <= 750:
        return 5
    elif w > 750 and w <= 950:
        return 6
    else:
        return 7

def click_to_add_mask(x, y):

    global mask
    global image

    coord = [[x, y]]

    pred_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(pred_image)
    input_point = np.array(coord)

    pred_masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True, )
    idx_scores = 0
    pred_mask = get_mask(pred_masks[idx_scores])
    
    print(scores)
    #print(type(scores))
    #print(masks.shape)

    if debug:
        plt.figure()
        plt.imshow(mask)
        plt.figure()
        plt.imshow(pred_mask)

    mask = add_mask(mask, pred_mask)

    if debug:
        plt.figure()
        plt.imshow(mask)
        plt.show()


def draw_to_add_mask(x, y):

    global draw_point_coord
    global clone
    global mask

    coord = (x, y)

    if len(draw_point_coord) < 3:

        cv2.circle(clone, coord, radius=1, color=(255, 0, 0), thickness=-1)
        draw_point_coord.append(coord)

    elif len(draw_point_coord) == 3:
        
        color = (255, 255, 255)
        thickness = determine_thickness(clone)

        cv2.line(mask, draw_point_coord[0], draw_point_coord[1], color, thickness)
        cv2.line(mask, draw_point_coord[1], draw_point_coord[2], color, thickness)

        draw_point_coord.append(0)

def erase_mask(x, y):

    global erase_point_coord
    global clone
    global mask

    coord = (x, y)

    if len(erase_point_coord) < 2:

        cv2.circle(clone, coord, radius=1, color=(0, 255, 0), thickness=-1)
        erase_point_coord.append(coord)

    elif len(erase_point_coord) == 2:

        mask_color = (0, 0, 0)
        rectangle_color = (0, 255, 0)

        cv2.rectangle(clone, erase_point_coord[0], erase_point_coord[1], rectangle_color, thickness=2)
        cv2.rectangle(mask, erase_point_coord[0], erase_point_coord[1], mask_color, thickness=-1)

        erase_point_coord.append(0)



def click_event(event, x, y, flags, params):
  
    global clone, image
    global mask, mask_color
    global result
    global option
    global draw_point_coord, erase_point_coord
    global cache
    global player_bbox 

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        print(x, ',', y)

        if option == 'Click':
            click_to_add_mask(x, y)

        elif option == 'Draw':
            draw_to_add_mask(x, y)

        elif option == 'Erase':

            erase_mask(x, y)

        else:
            pass

        result = apply_mask_to_image(clone, mask, mask_color)
        cv2.imshow(window_name, result)

    elif event == cv2.EVENT_RBUTTONDOWN:

        #print('right button')
        clone = image.copy()

        #mask = np.zeros_like(clone)
        mask = cache.copy()
        draw_point_coord = []
        erase_point_coord = []

        result = apply_mask_to_image(clone, mask, mask_color)
        cv2.imshow(window_name, result)
        #cv2.imshow(window_name, clone)

window_name = 'image'

# create output folder
mask_out_dir = os.path.join(output_dir, "stick_mask")

if not os.path.exists(mask_out_dir):
   # Create a new directory because it does not exist
   os.makedirs(mask_out_dir)
   print("The new directory 1 is created!")

image_dir = os.path.join(output_dir, "images")
if not os.path.exists(image_dir):
   # Create a new directory because it does not exist
   os.makedirs(image_dir)
   print("The new directory 1 is created!")


idx = 0

while idx < (len(image_path_list) - 1):

    image = cv2.imread(image_path_list[idx])
    name = os.path.basename(image_path_list[idx]).split('.')[0]

    window_title = window_name + ':' + name

    player_bbox_mask = cv2.imread(player_bbox_path_list[idx], cv2.IMREAD_GRAYSCALE)
    player_bbox = mask_to_bbox(player_bbox_mask)
    
    clone = image.copy()
    for bbox in player_bbox:
        cv2.rectangle(clone, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    
    color = (0, 0, 255)
    mask_color = create_color_hightlight(clone, color)

    img_name = name + '.jpg'
    img_save_path = os.path.join(image_dir , img_name)

    mask_name = name + '.png'
    mask_save_path = os.path.join(mask_out_dir , mask_name)

    if os.path.isfile(mask_save_path):
        mask = cv2.imread(mask_save_path)
    else:
        mask = np.zeros_like(clone).astype("uint8")

    cache = mask.copy()
    draw_point_coord = []
    erase_point_coord = []

    if close:
        break

    while True:

        result = apply_mask_to_image(clone, mask, mask_color)

        cv2.imshow(window_name, result)
        cv2.setWindowTitle(window_name, window_title)
        #cv2.imshow(window_name, mask)

        cv2.setMouseCallback(window_name, click_event)
        press_key = cv2.waitKey(0) & 0xFF

        if press_key == ord('q'):
            cv2.destroyAllWindows()
            close = True
            break

        elif press_key == ord('s'):

            idx += 1
            print('Skip')
            break

        elif press_key == ord('n'):

            cv2.imwrite(img_save_path, image)
            cv2.imwrite(mask_save_path, mask)
            print('Saved')
            idx += 1
            break

        elif press_key == ord('b'):

            idx -= 1

            #print(idx)
            if idx < 0:
                idx = 0

            break

        elif press_key == ord('r'):

            clone = image.copy()
            for bbox in player_bbox:
                cv2.rectangle(clone, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
            
            mask = np.zeros_like(clone)
            cache = mask.copy()
            
            draw_point_coord = []

            result = apply_mask_to_image(clone, mask, mask_color)
            cv2.imshow(window_name, result)

            print('Redo')

        elif press_key == ord('a'):

            option = 'Click'

            clone = image.copy()
            cache = mask.copy()

        elif press_key == ord('d'):

            option = 'Draw'
            draw_point_coord = []

            clone = image.copy()
            cache = mask.copy()

        elif press_key == ord('e'):

            option = 'Erase'
            erase_point_coord = []

            clone = image.copy()
            cache = mask.copy()

        elif press_key == ord('v'):

            option = 'View'

            clone = image.copy()
            for bbox in player_bbox:
                cv2.rectangle(clone, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)

        elif press_key == ord('j'):

            idx += 10
            break

    #garbage collection
    draw_point_coord = []
    erase_point_coord = []