import numpy as np
import cv2
from scipy.ndimage.measurements import label

import features as F

OVERLAP_RATIO = 0.8
PROBABILITY_THRESHOLD = 0.75
HEATMAP_THRESHOLD = 2

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def getWindows(img):
    windows = []
    image_length = img.shape[0] # 720
    # Large window over bottom half of image
    w = slide_window(img, x_start_stop=[None, None], 
                     y_start_stop=[np.int(image_length*0.5), image_length], 
                     xy_window=(256, 256), xy_overlap=(OVERLAP_RATIO, OVERLAP_RATIO))
    windows.append(w)
#    print("l:", len(w))

    # Medium window over bottom half of image
    w = slide_window(img, x_start_stop=[None, None], 
                     y_start_stop=[np.int(image_length*0.5), np.int(image_length*0.85)], 
                     xy_window=(128, 128), xy_overlap=(OVERLAP_RATIO, OVERLAP_RATIO))
    windows.append(w)
#    print("m:", len(w))

    # Small window over bottom half of image
    w = slide_window(img, x_start_stop=[None, None], 
                     y_start_stop=[np.int(image_length*0.5), np.int(image_length*0.70)], 
                     xy_window=(64, 64), xy_overlap=(OVERLAP_RATIO, OVERLAP_RATIO))
    windows.append(w)
#    print("s:", len(w))

    return np.concatenate(windows)

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Draw the surrounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates        
#         print(bbox)
        cv2.rectangle(imcopy, (bbox[0][0], bbox[0][1]), (bbox[1][0],bbox[1][1]), color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a single function that can extract features and make predictions
def find_cars(img, windows, clf, X_scaler):
    possible_cars = []
    f = F.Features()
    # Iterate over all windows in the list
    positive_predictions = 0
    positive_predictions_above_threshold = 0
    for w in windows:
        # Extract the test window from original image
        sub_image = img[w[0][1]:w[1][1], w[0][0]:w[1][0]]
        test_img = cv2.resize(sub_image, (64, 64))
        # Extract features for that window 
        features = f.extract_features_image(test_img, color_space='YCrCb')
        # Scale extracted features to be fed to classifier
        test_features = X_scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            # Confidence score
            confidence_score = clf.predict_proba(test_features)
#             print("confidence_score", confidence_score[0][1])
            if confidence_score[0][1] > PROBABILITY_THRESHOLD:   # probability threshold for positive detection
                possible_cars.append(w)
                positive_predictions_above_threshold += 1
            positive_predictions += 1
    # Return windows for positive detections
    print("positive_predictions:", positive_predictions)
    print("positive_predictions_above_threshold:", positive_predictions_above_threshold)
    return possible_cars

def add_heat(heatmap, bbox_list, expand_area=25):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], (box[0][0]-expand_area):(box[1][0]+expand_area)] += 1

    # Return updated heatmap
    return heatmap # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    # Return the image
    return img

def get_labeled_bboxes(labels):
    # Iterate through all detected cars
    cars = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cars.append(bbox)
    # Return the image
    return cars

# Define a single function that can extract features and make predictions
def harden_detections(image, possible_cars):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, possible_cars)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, HEATMAP_THRESHOLD)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    print("Labels:", labels[1])

    cars = get_labeled_bboxes(labels)
    
    return cars