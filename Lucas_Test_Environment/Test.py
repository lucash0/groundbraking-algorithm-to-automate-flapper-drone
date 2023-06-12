import cv2
from matplotlib import pyplot as plt

# read image
img = cv2.imread('../Test_Data/olwvvpruij/noUSB14.png', cv2.IMREAD_UNCHANGED)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# get dimensions of image
dimensions = img.shape

# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
# channels = img.shape[2]

print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)
# print('Number of Channels : ', channels)


# Input
x = 100
y = 100

if x< width//2:
    print('go left')
else:
    print('go right')

if y< height//2:
    print('go down')
else:
    print('go up')
