import cv2
import numpy as np
import glob
import os

class Daq_tool:
    def __init__(self, class_object, test_data_folder):
        self.class_object = class_object
        self.test_data_folder = test_data_folder


    def run_tool(self):
        self.process_images()
        self.user_input()

    def process_images(self):
        # Directory containing the images
        image_dir = self.test_data_folder

        # Retrieve all PNG files in the directory
        image_files = glob.glob(os.path.join(image_dir, '*.png'))

        # Sort the image files based on their number
        image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0][3:]))

        # List to store the processed images
        self.processed_images = []
        # List to store the parameters corresponding to each processed image
        self.processed_parameters = []

        # Create an instance of the JeVois class
        JeVois = self.class_object()
        # Process and store each image along with its parameters
        for i, image_file in enumerate(image_files):
            # Load the image
            loaded_image = cv2.imread(image_file)
            print(i, len(image_files))
            # if i == 300:
            #     break
            # Process the image using JeVois and retrieve the parameters
            processed_image, parameters = JeVois.process(i + 1, len(image_files), loaded_image)
            # Store the processed image and its parameters
            self.processed_images.append(processed_image)
            self.processed_parameters.append(parameters.copy())

    def user_input(self):
        # Prompt for user input
        print("Press 'm' to view images individually using arrow keys")
        print("Press 'v' to play the video continuously")
        print("Press 's' to save the video")
        choice = input("Enter your choice: ")

        if choice == 'm':
            # Display the images individually using arrow keys
            self.display_images(self.processed_images, self.processed_parameters, 0)
        elif choice == 'v':
            # Display the images as a video with 30fps
            self.display_images(self.processed_images, self.processed_parameters, int(1000 / 15))
        elif choice == 's':
            # Save the video
            self.display_images(self.processed_images, self.processed_parameters, int(1000 / 15), save_video=True)
        else:
            print("Invalid choice")

    def display_images(self, images, parameters, fps, save_video=False):
        frame_count = 0
        out = None
        delay = int(1000 / fps)  # Calculate the delay based on the desired FPS

        for i, img in enumerate(images):
            # Retrieve the parameters for the current frame
            params = parameters[i]

            # Create a blank white image with the same height as the input image
            params_img = 255 * np.ones((img.shape[0], 300, 3), dtype=np.uint8)

            # Add the parameters as text to the params_img
            for j, param in enumerate(params):
                text = f'Parameter {j + 1}: {param}'
                cv2.putText(params_img, text, (10, (j + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)

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
                    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                          (display_img.shape[1], display_img.shape[0]))

                # Write the frame to the video file
                out.write(display_img)

            frame_count += 1

        # Release the video writer if it was created
        if save_video and out is not None:
            out.release()

        cv2.destroyAllWindows()
