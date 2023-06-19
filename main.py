# Fortune favours the bold
from JeVois_code.v01 import Test_v1

import cv2
import glob
import numpy as np
import os

# Function to display images or video with parameters on the right-hand side
def display_images(images, parameters, delay, save_video=False):
    frame_count = 0
    out = None
    for img, params in zip(images, parameters):
        # Create a blank white image with the same height as the input image
        params_img = 255 * np.ones((img.shape[0], 300, 3), dtype=np.uint8)

        # Add the parameters as text to the params_img
        for i, param in enumerate(params):
            text = f'Parameter {i+1}: {param}'
            cv2.putText(params_img, text, (10, (i+1)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Concatenate the input image and params_img horizontally
        display_img = np.concatenate((img, params_img), axis=1)

        cv2.imshow('Image with Parameters', display_img)
        key = cv2.waitKey(delay)

        # Break the loop if the 'q' key is pressed
        if key == ord('q'):
            break

        # Break the loop if the 'm' key is pressed to switch to manual mode
        if key == ord('m'):
            break

        # Break the loop if the 'v' key is pressed to switch to video mode
        if key == ord('v'):
            break

        # Save the video if save_video flag is set
        if save_video:
            if frame_count == 0:
                # Create a VideoWriter object
                out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (display_img.shape[1], display_img.shape[0]))

            # Write the frame to the video file
            out.write(display_img)

        frame_count += 1

    # Release the video writer if it was created
    if save_video and out is not None:
        out.release()

    cv2.destroyAllWindows()

# Directory containing the images
image_dir = 'Test_Data\\test6\\'

# Retrieve all PNG files in the directory
image_files = glob.glob(os.path.join(image_dir, '*.png'))

# Sort the image files based on their number
image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0][3:]))

# List to store the processed images
processed_images = []
# List to store the parameters corresponding to each processed image
processed_parameters = []

# Create an instance of the JeVois class
JeVois = Test_v1(True)

# Process and store each image along with its parameters
for image_file in image_files:
    # Load the image
    loaded_image = cv2.imread(image_file)

    # Process the image using JeVois and retrieve the parameters
    processed_image, parameters = JeVois.process(1, 1, loaded_image)

    # Store the processed image and its parameters
    processed_images.append(processed_image)
    processed_parameters.append(parameters)

# Prompt for user input
print("Press 'm' to view images individually using arrow keys")
print("Press 'v' to play the video continuously")
print("Press 's' to save the video")
choice = input("Enter your choice: ")

if choice == 'm':
    # Display the images individually using arrow keys
    display_images(processed_images, processed_parameters, 0)
elif choice == 'v':
    # Display the images as a video with 30fps
    display_images(processed_images, processed_parameters, int(1000/30))
elif choice == 's':
    # Save the video
    display_images(processed_images, processed_parameters, int(1000/30), save_video=True)
else:
    print("Invalid choice")


if __name__ == "__main__":
    image_path = 'Test_Data/test6/img1.png'
    loaded_image = cv2.imread(image_path)

    JeVois = Test_v1(True)
    processed_image = JeVois.process(1, 1, loaded_image)

