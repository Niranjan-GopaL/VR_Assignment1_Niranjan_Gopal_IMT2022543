Include a README file describing how to run your code, the methods chosen,
results, and observations.
d. Ensure your visual outputs, including detection, segmentation, coin counts, and the
final panorama, are clearly labeled and included in the README file.

# About the Project

There are python scripts as required by the assignment guidelines and the more interactive and intuitive jupyter notebooks available for both of the tasks separately.


# Part 1 :  



# Part 2 : Image Stitching using openCV
Second part demonstrates an automated image stitching pipeline that combines overlapping photographs into seamless panoramic views. 
Using OpenCV's powerful computer vision capabilities, we can automatically detect matching points between images and blend them into a single cohesive photograph.
The goal of this project is to stitch two images (named “left.jpg” and “right.jpg”) together to construct a panorama image.   
Image stitching is  **the combination of images with overlapping sections to create a single panoramic or high-resolution image**.
Panoramic photography is a technique that combines multiple images from the same rotating camera to form a single, wide photo. 

o


##  Steps 🪜
1. Detect and match features.
2. Compute homography (perspective transform between frames) using **RANSAC** algorithm.
3. Warp one image onto the other perspective.
4. Combine the base and warped images while keeping track of the shift in origin.
5. Given the combination pattern, stitch multiple images.  


## Results 🚀

<p align="center">
<img src="https://github.com/sajmaru/Image-Stitching-OpenCV/blob/main/Keypoints.png" height = 200 width = 500>
</p>

<p align="center">
<img src="https://github.com/sajmaru/Image-Stitching-OpenCV/blob/main/Keypoints_Mapped.png" height = 200 width = 500>
</p>

<p align="center">
<img src="https://github.com/sajmaru/Image-Stitching-OpenCV/blob/main/Stitched%20Output.jpeg" height = 200 width = 500>
</p>

### 📚 Requirements

> - OpenCV 
> - Python 3.x
> - NumPy

### ⭐ Getting Started
1. Clone this repository
2. Place your images as left.jpg and right.jpg
3. Run the stitching script
4. View your panoramic masterpiece!



In the event of this error :-
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