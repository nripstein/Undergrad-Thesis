import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from pprint import pprint


# todo
# change it so instead of grabbing first frame, do first frame with 2 hands in it (low priority)
# add outlier detection for video lengths to get manual inspection (medium priority)


def extract_first_frame(video_filename: str) -> np.ndarray:
    """
    Extracts the first frame of an MP4 video and returns it as a NumPy array.

    Args:
        video_filename (str): The path to the input MP4 video file.

    Returns:
        np.ndarray: A NumPy array representing the first frame of the video.

    Raises:
        FileNotFoundError: If the video file does not exist.
        IOError: If the video file cannot be opened or the first frame cannot be read.
        Exception: If an unexpected error occurs during the process.

    Example:
        To extract the first frame from an MP4 video named "input_video.mp4":
        >>> first_frame = extract_first_frame("input_video.mp4")
    """
    try:
        # Check if the video file exists
        if not os.path.isfile(video_filename):
            raise FileNotFoundError(f"The video file '{video_filename}' does not exist.")

        # Open the video file
        video_capture = cv2.VideoCapture(video_filename)

        # Check if the video file was opened successfully
        if not video_capture.isOpened():
            raise IOError("Error: Could not open video file.")

        # Read the first frame from the video
        ret, frame = video_capture.read()

        # Check if a frame was successfully read
        if not ret:
            raise IOError("Error: Could not read the first frame.")

        # Release the video file
        video_capture.release()

        # Convert the frame to a NumPy array
        return np.asarray(frame)

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def display_image_with_matplotlib(image: np.ndarray, title: str = "Image") -> None:
    """
    Displays a NumPy image (array) using matplotlib.

    Args:
        image (numpy.ndarray): The image to be displayed (as a NumPy array).
        title (str): The title for the displayed image (default is "Image").

    Returns:
        None
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def detect_hands(image) -> tuple[np.ndarray, tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Detects hands in the input image and draws a bounding box around each detected hand,
    along with landmarks for each hand.

    Args:
        image (numpy.ndarray): The input image where hands are to be detected. It is assumed
            to be in BGR color format.

    Returns:
        tuple:
            numpy.ndarray: The output image with bounding boxes and landmarks drawn on it.
            tuple: the coordinates of form ((x_min, x_max), (y_min, y_max))
                defining the smallest box which surrounds both hands .

    Examples:
        >>> first_frame = extract_first_frame("path/to/video.mp4")
        >>> processed_image, box_bounds = detect_hands(first_frame)
        >>> display_image_with_matplotlib(processed_image, "Processed Image")

    Notes:
        - The input image should be in BGR color format, typically obtained using OpenCV.
        - The function uses the MediaPipe Hands solution for hand detection.
        - This function modifies the input image in-place. If the original image is needed,
          a copy should be created before calling this function.
        - If no hands are detected, the function returns the unmodified image and a result object
          with empty landmark fields.
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    mp_result = hands.process(image_rgb)

    x_output = []
    y_output = []

    if mp_result.multi_hand_landmarks:
        # print("in if")
        # Loop through each hand detected
        for hand_landmarks in mp_result.multi_hand_landmarks:
            # Find the bounding box coordinates
            x_coords = [lm.x * image.shape[1] for lm in hand_landmarks.landmark]
            y_coords = [lm.y * image.shape[0] for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw the hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(f"HAND DETECTED:\n"
                  f"x: ({x_min}, {x_max}), y: x in ({y_min}, {y_max})")

            # add boundary to output
            if not x_output:
                x_output = [x_min, x_max]
            else:
                if x_min < x_output[0]:
                    x_output[0] = x_min
                if x_max > x_output[1]:
                    x_output[1] = x_max

            if not y_output:
                y_output = [y_min, y_max]
            else:
                if y_min < y_output[0]:
                    y_output[0] = y_min
                if y_max > y_output[1]:
                    y_output[1] = y_max

    # close mediapipe hands API to free up resources
    hands.close()

    return image, (tuple(x_output), tuple(y_output))


def image_bounds_multiple_method(min_max_tuple: tuple, x_multiple: float = 1, y_multiple: float = 1.5) \
        -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Given coordinates of bounding box for hands, returns coordinates bounding box for whole image.
    Bounding box has padding around hands as determined by x_multiple and y_multiple.

    Args:
        min_max_tuple (tuple[tuple[int, int], tuple[int, int]]): bounding box around hand coordinates of form
            ((x_min, x_max), (y_min, y_max)). generated by detect_hands_and_draw_bbox()
        x_multiple (float): Multiple of hand bounding box length in x dimension to be padded around left and right of hands
        y_multiple (float): Multiple of hand bounding box length in y dimension to be padded around top and bottom of hands

    Returns:
        bounding box coordinates: ((x_min, x_max), (y_min, y_max)):
    """
    x = min_max_tuple[0]
    y = min_max_tuple[1]
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]
    # print(f"delta_x = {delta_x}, delta_y = {delta_y}")
    x_bounds = (int(x[0] - x_multiple * delta_x), int(x[1] + x_multiple * delta_x))
    y_bounds = (int(y[0] - y_multiple * delta_y), int(y[1] + y_multiple * delta_y))
    return x_bounds, y_bounds


def image_bounds_set_size_method(min_max_tuple: tuple, size_x: int = 652, size_y: int = 652) \
        -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Given coordinates of bounding box for hands, returns coordinates bounding box for whole image.
    Bounding box of predetermined size (size_x, size_y) centered around the hands

    Args:
        min_max_tuple (tuple[tuple[int, int], tuple[int, int]]): bounding box around hand coordinates of form
            ((x_min, x_max), (y_min, y_max)). generated by detect_hands_and_draw_bbox()
        size_x (int): x dimension of resulting bounding box
        size_y (int): y dimension of resulting bounding box

    Returns:
        bounding box coordinates: ((x_min, x_max), (y_min, y_max)):

    """
    x = min_max_tuple[0]
    y = min_max_tuple[1]
    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]

    x_center = x[0] + delta_x // 2
    y_center = y[0] + delta_y // 2

    # print(f"delta_x = {delta_x}, delta_y = {delta_y}")
    # print(f"x cent = {x_center}, y_cent = {y_center}")
    x_bounds = (x_center - size_x // 2, x_center + size_x // 2)
    y_bounds = (y_center - size_x // 2, y_center + size_y // 2)
    return x_bounds, y_bounds


def crop_video_no_white(video_filename: str, output_filename: str, crop_region: tuple) -> None:
    """
    Crops a video to a specified region. Raises error if specified region outside of video frame

    Args:
        video_filename (str): The path to the input video file.
        output_filename (str): The path to the output video file.
        crop_region (tuple): A tuple containing the pixel coordinates for the crop region
                             in the format ((min_x, max_x), (min_y, max_y)).

    Returns:
        None

    Raises:
        FileNotFoundError: If the input video file does not exist.
        ValueError: If the crop region is invalid.

    Example:
        >>> crop_video("input_video.mp4", "output_video.mp4", ((100, 500), (200, 600)))
    """
    if not os.path.isfile(video_filename):
        raise FileNotFoundError(f"The video file '{video_filename}' does not exist.")

    cap = cv2.VideoCapture(video_filename)  # open video

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    (min_x, max_x), (min_y, max_y) = crop_region

    if min_x < 0 or max_x > width or min_y < 0 or max_y > height or min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid crop region.")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # set export format to be mp4
    out = cv2.VideoWriter(output_filename,
                          fourcc,
                          fps,
                          (max_x - min_x, max_y - min_y))  # set parameters of video file we're writing to

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if i % 900 == 0:
        #     print(f"i = {i}, so {i / 30} seconds in")
        #     print(frame.shape)
            # original_y = frame.shape[0]
            # original_x = frame.shape[1]

        # Crop the frame
        cropped_frame = frame[min_y:max_y, min_x:max_x]

        # Write the cropped frame to the new video file
        out.write(cropped_frame)
        i += 1

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    print(f"The video has been cropped and saved to '{output_filename}'")


def crop_video1(video_filename: str, output_filename: str, crop_region: tuple) -> None:  # AUDIO SAVING NOT CHECKED
    """
       Crops a video to a specified region.  If crop region outside bounds, then white pixels are added

       Args:
           video_filename (str): The path to the input video file.
           output_filename (str): The path to the output video file.
           crop_region (tuple): A tuple containing the pixel coordinates for the crop region
                                in the format ((min_x, max_x), (min_y, max_y)).

       Returns:
           None

       Raises:
           FileNotFoundError: If the input video file does not exist.
           ValueError: If the crop region is invalid.

       Example:
           >>> crop_video("input_video.mp4", "output_video.mp4", ((100, 500), (200, 600)))
       """

    if not os.path.isfile(video_filename):
        raise FileNotFoundError(f"The video file '{video_filename}' does not exist.")

    cap = cv2.VideoCapture(video_filename)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    (min_x, max_x), (min_y, max_y) = crop_region

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, # cahnge to temp_filename for vid
                          fourcc,
                          fps,
                          (max_x - min_x, max_y - min_y))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a blank (white) frame of desired size
        blank_frame = np.ones((max_y - min_y, max_x - min_x, 3), dtype=np.uint8) * 255

        # Calculate the position to copy pixels from the source frame
        src_y1 = max(0, min_y)
        src_y2 = min(height, max_y)
        src_x1 = max(0, min_x)
        src_x2 = min(width, max_x)

        # Calculate the position to paste pixels in the blank frame
        dst_y1 = max(0, -min_y)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = max(0, -min_x)
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        # Copy pixels from the source frame to the blank frame
        blank_frame[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        # Write the frame with white padding to the new video file
        out.write(blank_frame)

    print(f"The video has been cropped and saved to '{output_filename}'")


def crop_video(video_filename: str, output_filename: str, crop_region: tuple) -> None:  # AUDIO SAVING NOT CHECKED
    """
       Crops a video to a specified region.  If crop region outside bounds, then white pixels are added

       Args:
           video_filename (str): The path to the input video file.
           output_filename (str): The path to the output video file.
           crop_region (tuple): A tuple containing the pixel coordinates for the crop region
                                in the format ((min_x, max_x), (min_y, max_y)).

       Returns:
           None

       Raises:
           FileNotFoundError: If the input video file does not exist.
           ValueError: If the crop region is invalid.

       Example:
           >>> crop_video("input_video.mp4", "output_video.mp4", ((100, 500), (200, 600)))
       """

    if not os.path.isfile(video_filename):
        raise FileNotFoundError(f"The video file '{video_filename}' does not exist.")

    if not os.path.isfile(video_filename):
        raise FileNotFoundError(f"The video file '{video_filename}' does not exist.")

    # import ffmpeg
    # temp_filename = "temp_video.mp4"

    cap = cv2.VideoCapture(video_filename)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    (min_x, max_x), (min_y, max_y) = crop_region

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename,  # change to "temp_filename" for vid
                          fourcc,
                          fps,
                          (max_x - min_x, max_y - min_y))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a blank (white) frame of desired size
        blank_frame = np.ones((max_y - min_y, max_x - min_x, 3), dtype=np.uint8) * 255

        # Calculate the position to copy pixels from the source frame
        src_y1 = max(0, min_y)
        src_y2 = min(height, max_y)
        src_x1 = max(0, min_x)
        src_x2 = min(width, max_x)

        # Calculate the position to paste pixels in the blank frame
        dst_y1 = max(0, -min_y)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = max(0, -min_x)
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        # Copy pixels from the source frame to the blank frame
        blank_frame[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        # Write the frame with white padding to the new video file
        out.write(blank_frame)

    # Adding audio from the original clip to the cropped video using ffmpeg-python
    # input_video = ffmpeg.input(temp_filename)
    # input_audio = ffmpeg.input(video_filename)

    # ffmpeg.output(input_video.video, input_audio.audio, output_filename).run()

    # Remove the temporary video file
    # os.remove(temp_filename)

    print(f"The video has been cropped and saved to '{output_filename}'")


# def zoom_video(input_file_path, crop_region, output_file_path):
#     """
#     Zooms the video to the specified crop region while retaining the original audio.
#
#     :param input_file_path: str, path to the input video file.
#     :param crop_region: tuple, pixel coordinates for the crop region in the format ((min_x, max_x), (min_y, max_y)).
#     :param output_file_path: str, path to save the output zoomed video file.
#     """
#     # os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
#     # from moviepy.editor import VideoFileClip
#     import ffmpeg
#
#
#     input_video = ffmpeg.input(input_file_path)
#     print("done vid")
#     input_audio = ffmpeg.input(input_file_path)
#
#     print(type(input_video), type(input_audio))
#
#     print("done aud")
#     ffmpeg.concat(input_video, input_audio, v=1, a=1).output('./processed_folder/finished_video.mp4').run()
#     print("done conc")
#
#     # Load the video clip
#     clip = VideoFileClip(input_file_path)
#
#     # Extract the crop region coordinates
#     min_x, max_x = crop_region[0]
#     min_y, max_y = crop_region[1]
#
#     # Crop the video to the specified region
#     cropped_clip = clip.crop(x1=min_x, y1=min_y, x2=max_x, y2=max_y)
#
#     # Set the audio of the cropped clip as the original audio
#     cropped_clip = cropped_clip.set_audio(clip.audio)
#
#     # Write the zoomed video file to the specified output path
#     cropped_clip.write_videofile(output_file_path, codec='libx264', audio_codec='aac')


def crop_vid_full(vid_name: str, output_name: str) -> None:
    """
    ONLY FUNCTION IN THIS FILE FOR EXTERNAL USE\n
    Given the name of a video, crops it

    Args:
        vid_name (str): The path to the input video file.
        output_name (str): The path to the output video file.

    Returns:
        None
    """
    first_frame = extract_first_frame(vid_name)
    processed_image, hand_box_boundary = detect_hands(first_frame)
    bounds_sized = image_bounds_set_size_method(hand_box_boundary)
    # bounds_sized = image_bounds_multiple_method(hand_box_boundary)

    # display_image_with_matplotlib(cv2.rectangle(processed_image.copy(),
    #                                             pt1=(bounds_sized[0][0], bounds_sized[1][0]),
    #                                             pt2=(bounds_sized[0][1], bounds_sized[1][1]),
    #                                             color=(255, 0, 255), thickness=2),
    #                               f"Bound box ({(bounds_sized[0][0], bounds_sized[1][0])}), ({bounds_sized[0][1], bounds_sized[1][1]})")
    # import time
    # time.sleep(5)
    crop_video(vid_name, output_name, bounds_sized)
    print(f"")


############################################# NOW LENGTH #############################################

# Define the lower and upper bounds of the blue color in HSV
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])


def load_cv_vid(filepath: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(filepath)


def count_blue_pixels_in_video(cap: cv2.VideoCapture, lower_blue: np.ndarray, upper_blue: np.ndarray) -> np.ndarray:
    """
    Counts the number of blue pixels in each frame of a video.

    Args:
        cap (cv2.VideoCapture): The video capture object.
        lower_blue (np.ndarray): The lower bound of the blue color in HSV.
        upper_blue (np.ndarray): The upper bound of the blue color in HSV.

    Returns:
        np.ndarray: A 1D array containing the count of blue pixels for each frame.

    Raises:
        IOError
    """

    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        raise ValueError("Video file not opened")

    # Initialize an empty list to store the count of blue pixels for each frame
    blue_pixel_counts = []

    actual_frame_count = 0

    # Loop through the video frames
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If the frame was not grabbed, then we have reached the end of the video
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a binary mask for the blue color
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Count the number of non-zero pixels in the mask (blue pixels in the original frame)
        num_blue_pixels = cv2.countNonZero(blue_mask)

        # Append the count to the list
        blue_pixel_counts.append(num_blue_pixels)

        actual_frame_count += 1

    # Release the video capture object
    # cap.release()
    # print(f"Actual frame count processed: {actual_frame_count}")
    # Convert the list of blue pixel counts to a 1D NumPy array and return it
    return np.array(blue_pixel_counts)


def local_max_avg_threshold(arr, avg_multiple: float = 1.5, padding: int = 70) -> np.ndarray:
    """
    Find the indices of maximum values in intervals that exceed a specified multiple of the average value.

    Args:
        arr (np.ndarray): The input NumPy array containing the data.
        avg_multiple (float, optional): The multiple of the average value used as the threshold. Default is 1.5.
        padding (int, optional): The number of elements to consider as padding. If a new maximum is found
            within this range of a previous maximum, it will not be counted. Default is 0.

    Returns:
        np.ndarray: An array containing the indices of maximum values within intervals exceeding the threshold.
    """
    # Calculate the threshold as a multiple of the average
    threshold = avg_multiple * np.mean(arr)

    # Initialize variables to keep track of intervals
    start = None
    max_indices = []

    for i, value in enumerate(arr):
        if value > threshold:
            if start is None:
                start = i
            elif i == len(arr) - 1:
                # Handle the case when the interval extends to the end of the array
                end = i
                interval = arr[start:end + 1]
                max_index = start + np.argmax(interval)
                if not max_indices or max_index - max_indices[-1] > padding:
                    max_indices.append(max_index)
        elif start is not None:
            end = i - 1
            interval = arr[start:end + 1]
            max_index = start + np.argmax(interval)
            if not max_indices or max_index - max_indices[-1] > padding:
                max_indices.append(max_index)
            start = None

    return np.array(max_indices)


def split_video(cap: cv2.VideoCapture, split_frame_indices: np.ndarray, output_dir: str = "split_clips") -> None:
    """
    Splits a video into multiple segments based on the provided split frame indices.

    Args:
        cap (cv2.VideoCapture): Input video file.
        split_frame_indices (np.ndarray): 1D array containing the indices of the split frames.
        output_dir (str): Directory where the split videos will be saved. Defaults to "split_clips"

    Returns:
        None. Video segments are saved to the specified output directory.
    """
    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        raise ValueError("Video file not opened")

    # Get the codec information of the input video
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # specifying mp4v doesn't cause warning
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get characteristics of the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the start frame index for the first segment
    start_frame_idx = 0

    # Iterate over the split frame indices to create video segments
    for i, split_frame_idx in enumerate(np.append(split_frame_indices, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # Define the output video path for the current segment
        output_video_path = os.path.join(output_dir, f"segment_{i + 1}.mp4")

        # Initialize the video writer object
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Write frames to the output video segment
        for frame_idx in range(start_frame_idx, split_frame_idx):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        # Release the video writer object
        out.release()

        # Update the start frame index for the next segment
        start_frame_idx = split_frame_idx

    # Release the video capture object
    cap.release()
    print(f"Split video segments have been saved to {output_dir}")


def get_video_lengths(directory: str) -> dict:
    """
    Retrieves the duration (in seconds) of all video files within a specified directory.

    Walks through the provided directory and its subdirectories, identifies video files with
    common extensions (e.g., .mp4, .avi, .mkv), and calculates the duration of each video using OpenCV's
    cv2.VideoCapture module.

    Args:
        directory (str): The path to the directory containing video files.

    Returns:
        dict: A dictionary where keys are the full paths to video files and values are their respective
        durations in seconds.

    Raises:
        Exception: If there is an error while processing any video file, such as inability to open the file
        or retrieve its properties, an exception is raised with an error message.

    Example:
        If you have a directory '/path/to/your/video/directory' containing video files:

        >>> directory_path = "/path/to/your/video/directory"
        >>> video_lengths = get_video_lengths(directory_path)

        The 'video_lengths' dictionary will contain video file paths as keys and their durations in seconds as values.
        You can then iterate through the dictionary to access the information:

        >>> for video_path, duration in video_lengths.items():
        ...     print(f"{video_path}: {duration:.2f} seconds")
    """
    video_lengths = {}

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv')):
                video_path = os.path.join(root, filename)

                try:
                    cap = cv2.VideoCapture(video_path)

                    # Get the frame count and frame rate to calculate duration
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                    duration = frame_count / frame_rate

                    # Store the duration in the dictionary
                    video_lengths[video_path] = duration

                    # Release the video capture object
                    cap.release()
                except Exception as e:
                    print(f"Error while processing {video_path}: {str(e)}")

    return video_lengths


def full_length_split(vid_dir: str) -> None:
    """
        ONLY FUNCTION IN THIS FILE FOR EXTERNAL USE\n
        Given the name of a video, crops it

        Args:
            vid_dir (str): The path to the input video file.

        Returns:
            None
        """
    # determine where to split based on blue pixels (from blue latex glove)
    cap_out = cv2.VideoCapture(vid_dir)
    blue_pixel_counts = count_blue_pixels_in_video(cap=cap_out, lower_blue=LOWER_BLUE, upper_blue=UPPER_BLUE)
    max_indices = local_max_avg_threshold(blue_pixel_counts, avg_multiple=2, padding=70)
    cap_out.release()

    # split videos
    vid_name, _ = os.path.splitext(vid_dir)
    output_dir = vid_name + "_clips"

    cap_out = cv2.VideoCapture(vid_dir)
    split_video(cap_out,
                max_indices[1::2],  # every other index because hand enters and leaves once per desired split
                output_dir)
    cap_out.release()

    # inspect video lengths
    vid_lengths = get_video_lengths(output_dir)
    print(f"CLIP LENGTHS OF {vid_dir}:\n")
    pprint(vid_lengths)

    # extra stuff to analyze outliers eventually
    # durations = np.array([duration for _, duration in sorted(vid_lengths.items())][2:])
    #
    # mean = np.mean(durations)
    # std_dev = np.std(durations)
    #
    # # Plot the histogram
    # plt.hist(durations, bins=20)
    # plt.xlabel('Duration')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Durations')
    # plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean ({mean:.2f})')
    # plt.legend()
    # plt.show()
    #
    # # Print 1 SD from the mean and 2 SD from the mean
    # one_std_above_mean = mean + std_dev
    # two_std_above_mean = mean + 2 * std_dev
    #
    # print(f'1 SD from the mean: {one_std_above_mean:.2f}')
    # print(f'2 SD from the mean: {two_std_above_mean:.2f}')


if __name__ == '__main__':
    output_file = "full_pipeline_1/full_test_1.mp4"
    crop_vid_full("whole_thing_no_crop.mp4", output_file)
    full_length_split(output_file)
