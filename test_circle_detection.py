from circle.CirclesFeature import CirclesFeature
import cv2
import numpy as np

image = cv2.imread('input_QA.jpg')
# Phase 1: Detect circles
circles = CirclesFeature.Detect(image)

# Phase 2: Circle box classification
for c in circles:
    r = c._radius
    x, y = c._centre.x, c._centre.y
    padding = 2
    box = image[y-r-padding: y+r+padding, x-r-padding:x+r+padding]
    
    if len(box.shape) == 3:
        box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

    # Rule
    mean_box = np.mean(box)
    if mean_box < 200:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    
    
    image = cv2.circle(image, (x, y), r, color, 2)
cv2.imwrite('output.jpg', image)

# Phase 3: ...
