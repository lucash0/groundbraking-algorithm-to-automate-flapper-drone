import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from random import randrange
import math

bgr = cv2.imread('Test_Data\Flight\img140.png')

# Convert the image to YUV color space
yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

h, w, d, = yuv.shape

# Define the lower and upper bounds of the orange color in YUV
lower_orange = np.array([0, 0, 130])  # Adjusted lower bound of orange color in YUV
upper_orange = np.array([255, 140, 255])  # Adjusted upper bound of orange color in YUV
max_samples = 1000
sigmaL = 60

highlight_color = (255, 255, 255)  # White color for highlighting
highlighted_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)  # Convert YUV to BGR for visualization


# Threshold the image to get a binary mask of the orange color
mask = cv2.inRange(yuv, lower_orange, upper_orange)

def isTargetColour(colour):
    if np.all(colour >= lower_orange) and np.all(colour <= upper_orange):
        return True

def searchUpDown(P0, yuv):
    x0 = P0[0]
    y0 = P0[1]
    x1 = x0
    y1 = y0
    x2 = x0
    y2 = y0
    done = False
    while done == False:
        if isTargetColour(yuv[y1-1, x1]):
            y1 = y1 - 1
        elif isTargetColour(yuv[y1-1, x1-1]):
            y1 = y1 - 1
            x1 = x1 - 1
        elif isTargetColour(yuv[y1-1, x1+1]):
            y1 = y1 - 1
            x1 = x1 + 1
        else:
            done = True
    done = False
    while done == False:
        if isTargetColour(yuv[y2+1, x2]):
            y2 = y2 + 1
        elif isTargetColour(yuv[y2+1, x2-1]):
            y2 = y2 + 1
            x2 = x2 - 1
        elif isTargetColour(yuv[y2+1, x2+1]):
            y2 = y2 + 1
            x2 = x2 + 1
        else:
            done = True
    P1 = (x1,y1)
    P2 = (x2,y2)
    return P1, P2, x1, y1


def searchLeftRight(P0, yuv):
    x0 = P0[0]
    y0 = P0[1]
    xl = x0
    yl = y0
    xr = x0
    yr = y0
    done = False
    while done == False:
        if isTargetColour(yuv[yl, xl-1]):
            xl = xl - 1
        elif isTargetColour(yuv[yl-1, xl-1]):
            yl = yl - 1
            xl = xl - 1
        elif isTargetColour(yuv[yl+1, xl-1]):
            yl = yl + 1
            xl = xl - 1
        else:
            done = True
    done = False
    while done == False:
        if isTargetColour(yuv[yr, xr+1]):
            xr = xr + 1
        elif isTargetColour(yuv[yr-1, xr+1]):
            yr = yr - 1
            xr = xr + 1
        elif isTargetColour(yuv[yr+1, xr+1]):
            yr = yr + 1
            xr = xr + 1
        else:
            done = True
    Pl = (xl, yl)
    Pr = (xr, yr)
    dist_Pl = math.dist(P0, Pl)
    dist_Pr = math.dist(P0, Pr)
    if dist_Pl > dist_Pr:
        PN = Pl
    else:
        PN = Pr
    return PN

detectedGates = []
for i in range(0, max_samples):
    #print(i)
    x0 = randrange(2, w - 3)
    y0 = randrange(4, h - 2)
    P0 = (x0,y0)
    colour0 = yuv[y0, x0]
    if isTargetColour(yuv[y0, x0]):
        #highlighted_image[y0, x0] = highlight_color
        P1, P2 = searchUpDown(P0, yuv)[:2]
        #print('P1:',P1,'P2:',P2)
        if math.dist(P1,P2) > sigmaL:
            P3 = searchLeftRight(P1, yuv)
            #print('P3:',P3)
            P4 = searchLeftRight(P2, yuv)
            #print('P4:',P4)
            if math.dist(P1,P3) > sigmaL and math.dist(P2,P4) > sigmaL: #change and to or when addding refinement filter
                detectedGate = [P1,P2,P3,P4]
                print('P1:', P1, 'P2:', P2)
                print('P3:', P3, 'P4:', P4)
                detectedGates.append(detectedGate)


print(detectedGates)
# Assuming you have the gates stored in the 'detected_gates' list

# Create a figure and axes for plotting
fig, ax = plt.subplots()

# Loop over the detected gates
for gate in detectedGates:
    # Extract the four corner points of the gate
    P1, P2, P3, P4 = gate

    # Create an array of points in the desired sequence
    points = np.array([P1, P2, P4, P3])

    # Draw the polygon on the image plot
    polygon = plt.Polygon(points, closed=True, edgecolor='g', facecolor='none')
    ax.add_patch(polygon)

    # Plot the pixel coordinates
    for point in points:
        ax.text(point[0], point[1], f'({point[0]}, {point[1]})', color='r')

# Display the image plot with polygons and pixel coordinates
ax.imshow(highlighted_image)
plt.show()

