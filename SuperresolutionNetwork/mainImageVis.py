"""
Utility to generate the "lenses" in the images in the paper
"""

import os
import os.path
import numpy as np
import cv2 as cv
import imageio

##########################
# CONFIG
##########################

CROP = (32, 32, 32, 32) # global outer crop
LINE_COLOR = 0
LINE_SIZE = 5
ALTERNATIVE_LINE = 0

if 0: #EJECTA
    INPUT_FILE = "screenshots/final/Ejecta-512/snapshot_272_512_ushort.nearest.color.03m28d-10h12m13s.png"
    OUTPUT_FILE = "screenshots/final/Ejecta-512/nearest.png"
    #INPUT_FILE = "screenshots/final/Ejecta-512/snapshot_272_512_ushort.gt.color.03m28d-10h14m05s.png"
    #OUTPUT_FILE = "screenshots/final/Ejecta-512/gt.png"
    #INPUT_FILE = "screenshots/final/Ejecta-512/snapshot_272_512_ushort.gen_l1normal.color.03m28d-10h14m49s.png"
    #OUTPUT_FILE = "screenshots/final/Ejecta-512/est.png"
    
    SNIPPET = (768, 240, 128, 128) # x, y, w, h
    PASTE = (40, 408, 4) # x, y, scale

elif 0: #SKULL
    PATH = "../Paper/figures/final/vmhead/"
    if 0:
        INPUT_FILE = PATH + "input.png"
        OUTPUT_FILE = PATH + "input_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "bilinear.png"
        OUTPUT_FILE = PATH + "bilinear_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "bicubic.png"
        OUTPUT_FILE = PATH + "bicubic_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "ours.png"
        OUTPUT_FILE = PATH + "ours_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "gt.png"
        OUTPUT_FILE = PATH + "gt_lens.jpg"

    SNIPPET = (540, 540, 128, 128)
    PASTE = (728, 40, 4)

elif 0: #Cleveland
    PATH = "../Paper/figures/final/cleveland70/"
    if 0:
        INPUT_FILE = PATH + "input.png"
        OUTPUT_FILE = PATH + "input_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "bilinear.png"
        OUTPUT_FILE = PATH + "bilinear_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "bicubic.png"
        OUTPUT_FILE = PATH + "bicubic_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "ours.png"
        OUTPUT_FILE = PATH + "ours_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "gt.png"
        OUTPUT_FILE = PATH + "gt_lens.jpg"

    SNIPPET = (768, 304, 128, 128)
    PASTE = (40, 408, 4)

elif 0: #Meshkov
    PATH = "../Paper/figures/final/RichtmyerMeshkov1024/"
    if 0:
        INPUT_FILE = PATH + "input.png"
        OUTPUT_FILE = PATH + "input_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "bilinear.png"
        OUTPUT_FILE = PATH + "bilinear_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "bicubic.png"
        OUTPUT_FILE = PATH + "bicubic_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "ours.png"
        OUTPUT_FILE = PATH + "ours_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "gt.png"
        OUTPUT_FILE = PATH + "gt_lens.jpg"

    SNIPPET = (640, 80, 128, 128)
    PASTE = (40, 408, 4)

elif 0: #Ejecta512_2
    PATH = "../Paper/figures/final/Ejecta512_2/"
    if 1:
        INPUT_FILE = PATH + "input.png"
        OUTPUT_FILE = PATH + "input_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "bilinear.png"
        OUTPUT_FILE = PATH + "bilinear_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "bicubic.png"
        OUTPUT_FILE = PATH + "bicubic_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "ours.png"
        OUTPUT_FILE = PATH + "ours_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "gt.png"
        OUTPUT_FILE = PATH + "gt_lens.jpg"

    SNIPPET = (450, 240, 128, 128)
    PASTE = (730, 270, 4)
    ALTERNATIVE_LINE = 2
    CROP = (32, 32, 170, 32) # global outer crop

elif 0: #Ejecta 1024_2
    CROP = (0, 1, 0, 1)
    PATH = "../Paper/figures/final/Ejecta1024_2/"
    if 0:
        INPUT_FILE = PATH + "nearest.png"
        OUTPUT_FILE = PATH + "nearest_lens.jpg"
    elif 0:
        INPUT_FILE = PATH + "ours.png"
        OUTPUT_FILE = PATH + "ours_lens.jpg"
    elif 1:
        INPUT_FILE = PATH + "gt.png"
        OUTPUT_FILE = PATH + "gt_lens.jpg"

    SNIPPET = (840, 240, 128, 128)
    PASTE = (40, 408, 4)

elif 1: #Ejecta 2
    CROP = (0, 1, 0, 1)
    PATH = "D:/VolumeSuperResolution/Video/images/Ejecta_old/"
    if 0:
        INPUT_FILE = PATH + "nearest_color.png"
        OUTPUT_FILE = PATH + "nearest_color_lens.png"
    elif 0:
        INPUT_FILE = PATH + "nearest_depth.png"
        OUTPUT_FILE = PATH + "nearest_depth_lens.png"
        LINE_COLOR = 120
    elif 1:
        INPUT_FILE = PATH + "fakeGT_depth.png"
        OUTPUT_FILE = PATH + "fakeGT_depth_lens.png"
        LINE_COLOR = 120

    SNIPPET = (720, 112, 128, 128)
    PASTE = (40, 408, 4)

elif 0: #Ejecta 2
    CROP = (0, 1, 0, 1)
    PATH = "D:/VolumeSuperResolution/Video/images/Ejecta/"
    if 0:
        INPUT_FILE = PATH + "nearest_color.png"
        OUTPUT_FILE = PATH + "nearest_color_lens.png"
    elif 0:
        INPUT_FILE = PATH + "l1_color.png"
        OUTPUT_FILE = PATH + "l1_color_lens.png"
    elif 0:
        INPUT_FILE = PATH + "l1_color_noAO.png"
        OUTPUT_FILE = PATH + "l1_color_noAO_lens.png"
    elif 0:
        INPUT_FILE = PATH + "gt_color_ao.png"
        OUTPUT_FILE = PATH + "gt_color_ao_lens.png"
    elif 0:
        INPUT_FILE = PATH + "gt_color_noAO.png"
        OUTPUT_FILE = PATH + "gt_color_noAO_lens.png"
    elif 0:
        INPUT_FILE = PATH + "shaded_color.png"
        OUTPUT_FILE = PATH + "shaded_color_lens.png"
    elif 0:
        INPUT_FILE = PATH + "nearest_normal.png"
        OUTPUT_FILE = PATH + "nearest_normal_lens.png"
    elif 0:
        INPUT_FILE = PATH + "l1_normal.png"
        OUTPUT_FILE = PATH + "l1_normal_lens.png"
    elif 0:
        INPUT_FILE = PATH + "nearest_mask.png"
        OUTPUT_FILE = PATH + "nearest_mask_lens.png"
        LINE_COLOR = 120
    elif 0:
        INPUT_FILE = PATH + "l1_mask.png"
        OUTPUT_FILE = PATH + "l1_mask_lens.png"
        LINE_COLOR = 120
    elif 0:
        INPUT_FILE = PATH + "nearest_depth.png"
        OUTPUT_FILE = PATH + "nearest_depth_lens.png"
        LINE_COLOR = 120
    elif 0:
        INPUT_FILE = PATH + "l1_depth.png"
        OUTPUT_FILE = PATH + "l1_depth_lens.png"
        LINE_COLOR = 120
    elif 1:
        INPUT_FILE = PATH + "l1_ao.png"
        OUTPUT_FILE = PATH + "l1_ao_lens.png"

    SNIPPET = (720, 112, 128, 128)
    PASTE = (40, 408, 4)

elif 0: #Clouds 1
    CROP = (0, 1, 0, 1)
    PATH = "screenshots/shaded/"
    if 0:
        INPUT_FILE = PATH + "cloud1-input.png"
        OUTPUT_FILE = PATH + "cloud1-input-lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud1-gt.png"
        OUTPUT_FILE = PATH + "cluod1-gt-lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud1-shaded.png"
        OUTPUT_FILE = PATH + "cloud1-shaded-lens.png"
    elif 1:
        INPUT_FILE = PATH + "cloud1-unshaded.png"
        OUTPUT_FILE = PATH + "cloud1-unshaded-lens.png"

    SNIPPET = (136, 400, 128, 128)
    PASTE = (620, 236, 3)
    ALTERNATIVE_LINE = 1

elif 0: #Clouds 2
    CROP = (0, 1, 0, 1)
    PATH = "screenshots/shaded/"
    if 1:
        INPUT_FILE = PATH + "cloud2-input.png"
        OUTPUT_FILE = PATH + "cloud2-input-lens.png"
    elif 1:
        INPUT_FILE = PATH + "cloud2-gt.png"
        OUTPUT_FILE = PATH + "cluod2-gt-lens.png"
    elif 1:
        INPUT_FILE = PATH + "cloud2-shaded.png"
        OUTPUT_FILE = PATH + "cloud2-shaded-lens.png"
    elif 1:
        INPUT_FILE = PATH + "cloud2-unshaded.png"
        OUTPUT_FILE = PATH + "cloud2-unshaded-lens.png"

    SNIPPET = (224, 400, 128, 128)
    PASTE = (620, 236, 3)
    ALTERNATIVE_LINE = True

elif 0: #Ejecta
    CROP = (300, 112, 150, 32)
    PATH = "screenshots/shaded/"
    if 0:
        INPUT_FILE = PATH + "ejecta-input.png"
        OUTPUT_FILE = PATH + "ejecta-input-lens.png"
    elif 0:
        INPUT_FILE = PATH + "ejecta-gt.png"
        OUTPUT_FILE = PATH + "ejecta-gt-lens.png"
    elif 0:
        INPUT_FILE = PATH + "ejecta-shaded.png"
        OUTPUT_FILE = PATH + "ejecta-shaded-lens.png"
    elif 1:
        INPUT_FILE = PATH + "ejecta-unshaded.png"
        OUTPUT_FILE = PATH + "ejecta-unshaded-lens.png"

    SNIPPET = (784, 240, 128, 128)
    PASTE = (320, 488, 3)

elif 0: #RM-shaded
    CROP = (0, 1, 0, 1)
    PATH = "screenshots/shaded/"
    if 0:
        INPUT_FILE = PATH + "ppmt273_512.shadedGenerator_EnhanceNet_percp+tl2+tgan1.color.03m30d-14h07m30s.png"
        OUTPUT_FILE = PATH + "rm-shaded-lens.png"
    elif 1:
        INPUT_FILE = PATH + "ppmt273_512.gen_l1normal.color.03m30d-14h07m28s.png"
        OUTPUT_FILE = PATH + "rm-unshaded-lens.png"

    SNIPPET = (1040, 192, 128, 128)
    PASTE = (20, 428, 4)

elif 0: #Ejecta-1024
    CROP = (0, 1, 168, 1)
    PATH = "screenshots/final/Ejecta-1024/"
    if 0:
        INPUT_FILE = PATH + "snapshot_272_1024.nearest.color.03m30d-11h09m55s.png"
        OUTPUT_FILE = PATH + "nearest.png"
    elif 0:
        INPUT_FILE = PATH + "snapshot_272_1024.gt.color.03m30d-11h21m57s.png"
        OUTPUT_FILE = PATH + "gtNoAO.png"
    elif 0:
        INPUT_FILE = PATH + "snapshot_272_1024.gt.color.03m30d-11h11m02s.png"
        OUTPUT_FILE = PATH + "gtWithAO.png"
    elif 0:
        INPUT_FILE = PATH + "snapshot_272_1024.gen_l1normal.color.03m30d-11h10m25s.png"
        OUTPUT_FILE = PATH + "l1Normal.png"
    elif 0:
        INPUT_FILE = PATH + "snapshot_272_1024.gen_gan_2b_nAO_wCol.color.03m30d-11h10m16s.png"
        OUTPUT_FILE = PATH + "gan2.png"
    elif 1:
        INPUT_FILE = PATH + "snapshot_272_1024.gen_gan_1.color.03m30d-11h24m12s.png"
        OUTPUT_FILE = PATH + "gan1.png"
    elif 1:
        INPUT_FILE = PATH + "snapshot_272_1024.bilinear.color.03m30d-11h09m58s.png"
        OUTPUT_FILE = PATH + "bilinear.png"

    SNIPPET = (736, 240-168, 128, 128)
    PASTE = (740, 428-168, 4)
    ALTERNATIVE_LINE = 2

elif 0: #Ejecta - Transfer Learning
    PATH = "../Paper/figures/transfer_learning/"
    if 0:
        INPUT_FILE = PATH + "ao_gt.png"
        OUTPUT_FILE = PATH + "ao_gt_lens.png"
    elif 0:
        INPUT_FILE = PATH + "ao_l1clouds.png"
        OUTPUT_FILE = PATH + "ao_l1clouds_lens.png"
    elif 1:
        INPUT_FILE = PATH + "ao_l1ejecta.png"
        OUTPUT_FILE = PATH + "ao_l1ejecta_lens.png"

    SNIPPET = (684, 148, 128, 128)
    PASTE = (40, 960-128*4-40, 4)

elif 0: #Cloud - Transfer Learning
    PATH = "../Paper/figures/transfer_learning/"
    if 1:
        INPUT_FILE = PATH + "cloud-gt.png"
        OUTPUT_FILE = PATH + "cloud_gt_lens.png"
    elif 1:
        INPUT_FILE = PATH + "cloud-l1clouds.png"
        OUTPUT_FILE = PATH + "cloud_l1clouds_lens.png"
    elif 1:
        INPUT_FILE = PATH + "cloud-l1ejecta.png"
        OUTPUT_FILE = PATH + "cloud_l1ejecta_lens.png"

    SNIPPET = (684, 200, 128, 128)
    PASTE = (40, 960-128*4-40, 4)

elif 1: #Cloud - All networks
    PATH = "../Paper/figures/allNets/"
    if 0:
        INPUT_FILE = PATH + "cloud-gt.png"
        OUTPUT_FILE = PATH + "cloud-gt_lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud-nearest.png"
        OUTPUT_FILE = PATH + "cloud-nearest_lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud-bilinear.png"
        OUTPUT_FILE = PATH + "cloud-bilinear_lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud-bicubic.png"
        OUTPUT_FILE = PATH + "cloud-bicubic_lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud-l1normal.png"
        OUTPUT_FILE = PATH + "cloud-l1normal_lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud-l1color.png"
        OUTPUT_FILE = PATH + "cloud-l1color_lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud-perc.png"
        OUTPUT_FILE = PATH + "cloud-perc_lens.png"
    elif 0:
        INPUT_FILE = PATH + "cloud-gan.png"
        OUTPUT_FILE = PATH + "cloud-gan_lens.png"
    elif 1:
        INPUT_FILE = PATH + "cloud-shaded.png"
        OUTPUT_FILE = PATH + "cloud-shaded_lens.png"

    SNIPPET = (874, 350, 128, 128)
    PASTE = (40, 960-128*4-40, 4)

########################
# PROCESS
########################

image = imageio.imread(INPUT_FILE)[:,:,0:3]
image = image.transpose((1, 0, 2))
print(image.shape)
image = image[CROP[0]:-CROP[1], CROP[2]:-CROP[3], :]

# extract snippet
snippet = image[
    SNIPPET[0]-CROP[0]:SNIPPET[0]-CROP[0]+SNIPPET[2], 
    SNIPPET[1]-CROP[1]:SNIPPET[1]-CROP[1]+SNIPPET[3], 
    :]
snippet = cv.resize(snippet, dsize=None, fx=PASTE[2], fy=PASTE[2], 
                    interpolation=cv.INTER_NEAREST)

# paste snippet
image[
    PASTE[0]-CROP[0]:PASTE[0]-CROP[0]+PASTE[2]*SNIPPET[2], 
    PASTE[1]-CROP[1]:PASTE[1]-CROP[1]+PASTE[2]*SNIPPET[3], 
    #image.shape[1]-PASTE[1]+CROP[1]-PASTE[2]*SNIPPET[2]:image.shape[1]-PASTE[1]+CROP[1],
    :] = snippet

# draw lines
if ALTERNATIVE_LINE==0:
    image = cv.line(image, 
        (PASTE[1]-CROP[1], PASTE[0]-CROP[0]),
        (SNIPPET[1]-CROP[1], SNIPPET[0]-CROP[0]),
        (LINE_COLOR,LINE_COLOR,LINE_COLOR),
        thickness=LINE_SIZE,
        lineType=cv.LINE_AA).get()
    image = cv.line(image, 
           (PASTE[1]-CROP[1]+PASTE[2]*SNIPPET[3], PASTE[0]-CROP[0]+PASTE[2]*SNIPPET[2]),
           (SNIPPET[1]-CROP[1]+SNIPPET[3], SNIPPET[0]-CROP[0]+SNIPPET[2]),
           (LINE_COLOR,LINE_COLOR,LINE_COLOR),
           thickness=LINE_SIZE,
           lineType=cv.LINE_AA)
elif ALTERNATIVE_LINE==1:
    image = cv.line(image, 
        (PASTE[1]-CROP[1], PASTE[0]-CROP[0]),
        (SNIPPET[1]-CROP[1], SNIPPET[0]-CROP[0]),
        (LINE_COLOR,LINE_COLOR,LINE_COLOR),
        thickness=LINE_SIZE,
        lineType=cv.LINE_AA).get()
    image = cv.line(image, 
           (PASTE[1]-CROP[1]+PASTE[2]*SNIPPET[3], PASTE[0]-CROP[0]),
           (SNIPPET[1]-CROP[1]+SNIPPET[3], SNIPPET[0]-CROP[0]),
           (LINE_COLOR,LINE_COLOR,LINE_COLOR),
           thickness=LINE_SIZE,
           lineType=cv.LINE_AA)
elif ALTERNATIVE_LINE==2:
    image = cv.line(image, 
        (PASTE[1]-CROP[1]+PASTE[2]*SNIPPET[3], PASTE[0]-CROP[0]),
        (SNIPPET[1]-CROP[1]+SNIPPET[3], SNIPPET[0]-CROP[0]),
        (LINE_COLOR,LINE_COLOR,LINE_COLOR),
        thickness=LINE_SIZE,
        lineType=cv.LINE_AA).get()
    image = cv.line(image, 
           (PASTE[1]-CROP[1], PASTE[0]-CROP[0]+PASTE[2]*SNIPPET[2]),
           (SNIPPET[1]-CROP[1], SNIPPET[0]-CROP[0]+SNIPPET[2]),
           (LINE_COLOR,LINE_COLOR,LINE_COLOR),
           thickness=LINE_SIZE,
           lineType=cv.LINE_AA)

image = cv.rectangle(image, 
                     (SNIPPET[1]-CROP[1], SNIPPET[0]-CROP[0]),
                     (SNIPPET[1]-CROP[1]+SNIPPET[3], SNIPPET[0]-CROP[0]+SNIPPET[2]),
                     (LINE_COLOR,LINE_COLOR,LINE_COLOR),
                     thickness=LINE_SIZE)
image = cv.rectangle(image,
                     (PASTE[1]-CROP[1], PASTE[0]-CROP[0]),
                     (PASTE[1]-CROP[1]+PASTE[2]*SNIPPET[3], PASTE[0]-CROP[0]+PASTE[2]*SNIPPET[2]),
                     (LINE_COLOR,LINE_COLOR,LINE_COLOR),
                     thickness=LINE_SIZE)
#image2 = np.copy(image)
#cv.rectangle(image, (5, 5), (50, 50), (0,0,0), 4)
#print('difference: ', np.sum(image-image2))

# save
image = image.transpose((1, 0, 2))
imageio.imwrite(OUTPUT_FILE, image)
