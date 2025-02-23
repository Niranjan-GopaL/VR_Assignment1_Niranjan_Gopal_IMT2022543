import cv2
import numpy as np

def count_coin(img):
    image_copy = img.copy()
    img_blur = cv2.GaussianBlur(img, (7, 7), 10)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Calculate the area of each contour
    area = {}
    for i in range(len(contours)):
        cnt = contours[i]
        ar = cv2.contourArea(cnt)
        area[i] = ar

    # Sort contours by area in descending order
    srt = sorted(area.items(), key=lambda x: x[1], reverse=True)
    results = np.array(srt).astype("int")

    # Count the number of contours with an area greater than 500
    num = np.argwhere(results[:, 1] > 500).shape[0]

    # Draw contours with green color on the copied image
    for i in range(1, num):
        image_copy = cv2.drawContours(image_copy, contours, results[i, 0], (0, 255, 0), 3)

    # Print the number of coins
    print("Number of coins is", num - 1)


# image_path    = "./../img/coins.jpg"  
image_path    = "./../img/coins_redmi_3.jpeg"  # got all except 2 correct
# image_path    = "./../img/one_redmi.jpg"
# # image_path  = "./../img/two_redmi.jpg"
# # image_path  = "./../img/three_redmi.jpg"
# image_path    = "./../img/four_redmi.jpg"
# # image_path  = "./../img/five_redmi.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (720, 900)) # resize to a smaller resolution
count_coin(img)