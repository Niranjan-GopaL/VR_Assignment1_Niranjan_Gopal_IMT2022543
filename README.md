# About the Project
This report presents the implementation and analysis of two computer vision projects: (1) a coin detection and counting system utilizing image segmentation techniques, and (2) an automated image stitching pipeline for creating panoramic images. The first part focuses on detecting and counting coins in images using contour detection and area-based segmentation. The second part demonstrates the creation of panoramic images through feature matching, homography computation, RANSAC algorithm  and image warping. Both implementations showcase practical applications of computer vision algorithms while addressing real-world challenges in image processing.

> [!NOTE]
> There are python scripts as required by the assignment guidelines and the more interactive and intuitive jupyter notebooks available for both of the tasks separately.


# Part 1 :  Coin Detection, Segmentation and Count

The coin detection system employs OpenCV’s contour detection algorithm to identify circular objects within the image. 
The implementation follows these key steps :-
1. Convert the input image to grayscale
2. Apply threshold to create a binary image
3. Detect contours using cv2.findContours
4. Filter contours based on area to eliminate noise

The segmentation phase involves:
1. Computing contour areas
2. Sorting contours by area in descending order
3. Applying area threshold ( ≥ 500 pixels)
4. Creating unique color masks for each identified coin

### Detection and Count
<p align="center">
<img src="https://github.com/Niranjan-GopaL/VR_Assignment1_Niranjan_Gopal_IMT2022543/blob/main/Coin_Detect_Segment_Count/coins_detected.png?raw=true" height = 200 width = 500>
</p>

### Segmentaion
<p align="center">
<img src="https://github.com/Niranjan-GopaL/VR_Assignment1_Niranjan_Gopal_IMT2022543/blob/main/Coin_Detect_Segment_Count/segmented_coins.png?raw=true" height = 200 width = 500>
</p>



# Part 2 : Image Stitching using openCV
Second part demonstrates an automated image stitching pipeline that combines overlapping photographs into seamless panoramic views. 
Using OpenCV's powerful computer vision capabilities, we can automatically detect matching points between images and blend them into a single cohesive photograph.
The goal of this project is to stitch two images (named “left.jpg” and “right.jpg”) together to construct a panorama image.   
Image stitching is  **the combination of images with overlapping sections to create a single panoramic or high-resolution image**.
Panoramic photography is a technique that combines multiple images from the same rotating camera to form a single, wide photo. 

Overall Algorithm :- 
1. Detect and match features.
2. Compute homography (perspective transform between frames) using **RANSAC** algorithm.
3. Warp one image onto the other perspective.
4. Combine the base and warped images while keeping track of the shift in origin.
5. Given the combination pattern, stitch multiple images.  

## Detect descriptors in each of the Image 
<p align="center">
<img src="https://github.com/Niranjan-GopaL/VR_Assignment1_Niranjan_Gopal_IMT2022543/blob/main/Image-Stitching/keypoints.png?raw=true" height = 200 width = 500>
</p>

## Mathcing Points using Homography Matrix
<p align="center">
<img src="https://github.com/Niranjan-GopaL/VR_Assignment1_Niranjan_Gopal_IMT2022543/blob/main/Image-Stitching/matching_keypoints.png?raw=true" height = 200 width = 500>
</p>

## Inliers and Outliers after RANSAC algorithms
<p align="center">
<img src="https://github.com/Niranjan-GopaL/VR_Assignment1_Niranjan_Gopal_IMT2022543/blob/main/Image-Stitching/inliers_outliers.png?raw=true" height = 200 width = 500>
</p>


### Requirements :-
> - OpenCV 
> - Python 3.x
> - NumPy

### Usage :-
1. Clone this repository
2. Place images in img folder, and refernce in appropriately in the code
3. have the python environment with the above dependencies
3. Run the `coin_count.ipynb` or `panorama.ipynb` | You can use their counterpart py scripts too

While using panorama, in the event of this error :-
`error: OpenCV(4.10.0) /home/conda/feedstock_root/build_artifacts/libopencv_1735819861380/work/opencv_contrib/modules/xfeatures2d/src/surf.cpp:1026: error: (-213:The function/feature is not implemented) This algorithm is patented and is excluded in this configuration; Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library in function 'create'`
which is triggered by the lines this :-
```py
    key_points1 = sift.detect(l_img, None)
    key_points1, descriptor1 = surf.compute(l_img, key_points1)
```
Please replace it by :-
```py
    key_points1, descriptor1 = orb.detectAndCompute(l_img, None)
    key_points2, descriptor2 = orb.detectAndCompute(r_img, None)
```