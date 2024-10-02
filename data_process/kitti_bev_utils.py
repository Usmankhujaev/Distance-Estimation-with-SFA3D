"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
"""

import math
import os
import sys
import random as rand
import cv2
import numpy as np
import time

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf


# def makeBEVMap(PointCloud_, boundary):
#     Height = cnf.BEV_HEIGHT + 1
#     Width = cnf.BEV_WIDTH + 1

#     # Discretize Feature Map
#     PointCloud = np.copy(PointCloud_)
#     PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
#     PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

#     # sort-3times
#     sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
#     PointCloud = PointCloud[sorted_indices]
#     _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
#     PointCloud_top = PointCloud[unique_indices]

#     # Height Map, Intensity Map & Density Map
#     heightMap = np.zeros((Height, Width))
#     intensityMap = np.zeros((Height, Width))
#     densityMap = np.zeros((Height, Width))

#     # some important problem is image coordinate is (y,x), not (x,y)
#     max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
#     heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

#     normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
#     intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
#     densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

#     RGB_Map = np.zeros((3, Height - 1, Width - 1))
#     RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
#     RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
#     RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

#     return RGB_Map

def makeBEVMap(PointCloud_, boundary):
    start_time = time.time()  # Start time

    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1
    flops = 0

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    flops += PointCloud.size  # np.copy: copy elements
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    flops += PointCloud.shape[0]  # division
    flops += PointCloud.shape[0]  # floor
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)
    flops += PointCloud.shape[0]  # division
    flops += PointCloud.shape[0]  # floor
    flops += PointCloud.shape[0]  # addition

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    flops += PointCloud.shape[0] * np.log(PointCloud.shape[0])  # sorting
    PointCloud = PointCloud[sorted_indices]

    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    flops += PointCloud.shape[0] * np.log(PointCloud.shape[0])  # unique
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width), dtype=np.float32)
    intensityMap = np.zeros((Height, Width), dtype=np.float32)
    densityMap = np.zeros((Height, Width), dtype=np.float32)
    flops += Height * Width * 3  # initialization

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    flops += 1  # subtraction
    flops += 1  # abs
    flops += 1  # float conversion

    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height
    flops += PointCloud_top.shape[0]  # division

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    flops += unique_counts.size  # addition
    flops += unique_counts.size  # log
    flops += unique_counts.size  # division
    flops += unique_counts.size  # minimum
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    flops += 3 * (Height - 1) * (Width - 1)  # initialization
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Calculate processing time
    print(f"Processing time: {processing_time:.6f} seconds")
    print(f"Estimated FLOPs: {flops}")

    return RGB_Map
# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


def drawRotatedBox(img, x, y, w, l, yaw, color):
    
    # modifications by U.S.S
    #print(y)
    #bev_dis_y = (-18+y) / 12.16
    b = rand.randint(0,100)
    g = rand.randint(200,255)
    r = rand.randint(250,255)
    # end of modification
   
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)

    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2).astype(int)
    
    center_x = int(np.mean(bev_corners[:, 0]))
    center_y = int(np.mean(bev_corners[:, 1]))
    cv2.circle(img, (center_x, center_y), 2, (255, 0, 0), -1)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)
    cv2.arrowedLine(img, (center_x, center_y), (int((corners_int[0, 0]+corners_int[3,0])/2), int((corners_int[3,1]+corners_int[0, 1])/2)), color, 2, tipLength=0.1)
    centroid = (np.array([[center_y], [center_x]]))
    return centroid
    # Modifications by u.s.s
    #cv2.putText(img, text+'m' ,(int(x+10),int(y+10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (b,g,r), 1, cv2.LINE_AA)
import random 

def extract_image_patch(image, x, y, w, l, yaw, patch_size=(64, 64)):
    """
    Extracts an image patch centered around the detected object.

    :param image: The original image.
    :param x: The x-coordinate of the object.
    :param y: The y-coordinate of the object.
    :param w: The width of the object.
    :param l: The length of the object.
    :param yaw: The orientation of the object.
    :param patch_size: The size of the image patch to extract.
    :return: The extracted image patch.
    """
    # Draw the box and get the centroid
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    centroid = drawRotatedBox(image, x, y, w, l, yaw, color)[0]

    # Define the patch coordinates
    x_min = max(int(centroid[1] - patch_size[0] / 2), 0)
    y_min = max(int(centroid[0] - patch_size[1] / 2), 0)
    x_max = min(int(centroid[1] + patch_size[0] / 2), image.shape[1])
    y_max = min(int(centroid[0] + patch_size[1] / 2), image.shape[0])

    # Extract and return the patch
    return image[y_min:y_max, x_min:x_max]

