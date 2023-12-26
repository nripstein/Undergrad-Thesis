import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from tqdm import tqdm
import tempfile
import zipfile

# streamlit run streamlit_zoom/streamlit_preprocess.py --server.maxUploadSize 1024 # run this in terminal for local running

st.set_page_config(page_title="Data Preprocessing")  # Set tab title

# Hide hamburger menu and Streamlit watermark
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)


# --------------------------------------- Functions ---------------------------------------
def extract_frame_n_st(video_filename: str, n: int = 1) -> np.ndarray:
    """
    Extracts the first frame of an MP4 video and returns it as a NumPy array.

    Args:
        video_filename (str): Input file name
        n (int): which frame number to extract

    Returns:
        np.ndarray: A NumPy array representing the first frame of the video.

    Raises:
        FileNotFoundError: If the video file does not exist.
        IOError: If the video file cannot be opened or the first frame cannot be read.
        Exception: If an unexpected error occurs during the process.

    Example:
        To extract the third frame from an MP4 video named "input_video.mp4":
        >>> first_frame = extract_frame_n_st("input_video.mp4", 3)
    """
    if type(n) != int or n < 1:
        raise TypeError("Frame number must be a positive integer (must be < 1)")

    try:
        # Open the video file
        # tfile = tempfile.NamedTemporaryFile(delete=False)
        # tfile.write(video_file.read())
        # video_capture = cv2.VideoCapture(tfile.name)
        video_capture = cv2.VideoCapture(video_filename)
        # Check if the video file was opened successfully
        if not video_capture.isOpened():
            st.markdown("couldnt open")
            raise IOError("Error: Could not open video file.")

        # Read the first frame from the video
        for _ in range(n):
            ret, frame = video_capture.read()

        # Check if a frame was successfully read
        if not ret:
            st.markdown("couldnt open frame 1")
            raise IOError("Error: Could not read the first frame.")

        # Release the video file
        video_capture.release()
        # Convert the frame to a NumPy array
        return np.asarray(frame)

    except FileNotFoundError as e:
        st.markdown(f"FileNotFoundError: {str(e)}")
        print(f"FileNotFoundError: {str(e)}")
    except Exception as e:
        st.markdown(f"An error occurred: {str(e)}, {type(e)}")
        print(f"An error occurred: {str(e)}")


def detect_hands_and_draw_bbox_st(image) -> tuple[np.ndarray, NotImplemented, tuple[tuple[int, int], tuple[int, int]]]:
    """
    Detects hands in the input image and draws a bounding box around each detected hand,
    along with landmarks for each hand.

    Args:
        image (numpy.ndarray): The input image where hands are to be detected. It is assumed
            to be in BGR color format.

    Returns:
        tuple:
            numpy.ndarray: The output image with bounding boxes and landmarks drawn on it.
            mediapipe.python.solution_base.SolutionOutputs: The MediaPipe detection results,
                containing information about the detected hands and landmarks.

    Examples:
        >>> first_frame = extract_first_frame("path/to/video.mp4")
        >>> processed_image, detection_result = detect_hands_and_draw_bbox(first_frame)
        >>> display_image_with_matplotlib(processed_image, "Processed Image")

    Notes:
        - The input image should be in BGR color format, typically obtained using OpenCV.
        - The function uses the MediaPipe Hands solution for hand detection.
        - This function modifies the input image in-place. If the original image is needed,
          a copy should be created before calling this function.
        - If no hands are detected, the function returns the unmodified image and a result object
          with empty landmark fields.
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    mp_result = hands.process(image_rgb)

    x_output = []
    y_output = []

    if mp_result.multi_hand_landmarks:  # if there is at least one hand detected
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
    return image, mp_result, (tuple(x_output), tuple(y_output))


def image_bounds_set_size_method(min_max_tuple: tuple, size_x: int = 652, size_y: int = 652) -> tuple[
    tuple[int, int], tuple[int, int]]:
    """
    Given coordinates of bounding box for hands, returns coordinates bounding box for whole image. Bounding box of predetermined size (size_x, size_y) centered around the hands

    Args:
        min_max_tuple (tuple[tuple[int, int], tuple[int, int]]): bounding box around hand coordinates of form ((x_min, x_max), (y_min, y_max)). generated by detect_hands_and_draw_bbox()
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

    # print(f"deltax = {delta_x}, deltay = {delta_y}")
    # print(f"x cent = {x_center}, y_cent = {y_center}")
    x_bounds = (x_center - size_x // 2, x_center + size_x // 2)
    y_bounds = (y_center - size_x // 2, y_center + size_y // 2)
    return x_bounds, y_bounds


def crop_video(video_filename: str, output_filename: str, crop_region: tuple, vert_flip: bool = False) -> None:
    """
    Crops a video to a specified region.

    Args:
        video_filename (str): The path to the input video file.
        output_filename (str): The path to the output video file.
        crop_region (tuple): A tuple containing the pixel coordinates for the crop region
                             in the format ((min_x, max_x), (min_y, max_y)).
        vert_flip (bool): if True, then the video gets flipped vertically

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # for progress bar

    (min_x, max_x), (min_y, max_y) = crop_region

    if min_x < 0 or max_x > width or min_y < 0 or max_y > height or min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid crop region.")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # set export format to be mp4
    out = cv2.VideoWriter(output_filename, fourcc, fps,
                          (max_x - min_x, max_y - min_y))  # the video file we're writing to

    if vert_flip:
        desc = "Flipping and cropping video..."
    else:
        desc = "Cropping video..."

    # Initialize progress bar
    progress_bar = st.progress(0)
    current_frame = 0
    with tqdm(total=total_frames, desc=desc) as progress_bar_tqdm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Crop the frame
            cropped_frame = frame[min_y:max_y, min_x:max_x]

            if vert_flip:
                cropped_frame = cv2.flip(cropped_frame, 0)

            # Write the cropped frame to the new video file
            out.write(cropped_frame)
            current_frame += 1
            progress_bar_tqdm.update(1)  # Update the tqdm progress bar
            progress_percent = min(1.0, current_frame / total_frames)
            progress_bar.progress(progress_percent, f"{desc} {progress_percent*100:.2f}%")

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    print(f"{video_filename} has been cropped and saved to '{output_filename}'")
    # st.success(f"{video_filename} has been cropped and saved to '{output_filename}'")

    progress_bar.progress(100, "Cropping complete")


def crop_and_extract(video_filename: str, output_filename: str, crop_region: tuple, vert_flip: bool = False, suffix: str = None) -> list[str]:
    """
    Crops a video to a specified region. and extracts frames

    Args:
        video_filename (str): The path to the input video file.
        output_filename (str): The path to the output video file.
        crop_region (tuple): A tuple containing the pixel coordinates for the crop region
                             in the format ((min_x, max_x), (min_y, max_y)).
        vert_flip (bool): if True, then the video gets flipped vertically
        suffix (str): string containing suffix (excluding file extension) for frame file after frame number

    Returns:
        list of files it created

    Raises:
        FileNotFoundError: If the input video file does not exist.
        ValueError: If the crop region is invalid.

    Example:
        >>> crop_video("input_video.mp4", "output_video.mp4", ((100, 500), (200, 600)))
    """
    if not os.path.isfile(video_filename):
        raise FileNotFoundError(f"The video file '{video_filename}' does not exist.")

    if not os.path.exists(output_filename):
        os.makedirs(output_filename)

    cap = cv2.VideoCapture(video_filename)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # for progress bar

    (min_x, max_x), (min_y, max_y) = crop_region

    if min_x < 0 or max_x > width or min_y < 0 or max_y > height or min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid crop region.")

    if vert_flip:
        desc = "Extracting frames, flipping and cropping video"
    else:
        desc = "Extracting frames and cropping video"

    generated_files = []

    # Initialize progress bar
    progress_bar = st.progress(0)
    current_frame: int = 0
    with tqdm(total=total_frames, desc=desc) as progress_bar_tqdm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Crop the frame
            cropped_frame = frame[min_y:max_y, min_x:max_x]

            if vert_flip:
                cropped_frame = cv2.flip(cropped_frame, 0)

            frame_path = os.path.join(output_filename, f'{current_frame}_{suffix}.jpg')
            cv2.imwrite(frame_path, cropped_frame)  # Save the current frame as an image file.
            generated_files.append(frame_path)

            current_frame += 1
            progress_bar_tqdm.update(1)  # Update the tqdm progress bar
            progress_percent = min(1.0, current_frame / total_frames)
            progress_bar.progress(progress_percent, f"{desc} {progress_percent*100:.2f}%")

    # Release the VideoCapture and VideoWriter objects
    cap.release()

    print(f"{video_filename} has been cropped and extracted, saved to '{output_filename}'")
    # st.success(f"{video_filename} has been cropped and extracted, saved to '{output_filename}'")

    progress_bar.progress(100, "Cropping and extracting complete")
    return generated_files


def image_bounds_set_size_method(min_max_tuple: tuple, size_x: int = 652, size_y: int = 652) -> tuple[
    tuple[int, int], tuple[int, int]]:
    """
    Given coordinates of bounding box for hands, returns coordinates bounding box for whole image. Bounding box of predetermined size (size_x, size_y) centered around the hands

    Args:
        min_max_tuple (tuple[tuple[int, int], tuple[int, int]]): bounding box around hand coordinates of form ((x_min, x_max), (y_min, y_max)). generated by detect_hands_and_draw_bbox()
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

    # print(f"deltax = {delta_x}, deltay = {delta_y}")
    # print(f"x cent = {x_center}, y_cent = {y_center}")
    x_bounds = (x_center - size_x // 2, x_center + size_x // 2)
    y_bounds = (y_center - size_x // 2, y_center + size_y // 2)
    return x_bounds, y_bounds


@st.cache_data(show_spinner=False)
def frame_with_detected_hands(video_path: str, n: int = 2) -> tuple[int, int]:
    """
    Find the first frame in a video where n hands are identified using MediaPipe Hands.

    Args:
        video_path (str): Path to the video file.
        n (int): Number of hands to detect (default is 2).

    Returns:
        int: Frame number where n hands are first detected, and the number of hands detected (frame_num, n).
        If n hands not detected but there's a frame with one hand detected, returns (first_frame_w_one, 1).
        returns (-1, 0) if no hands found.
    """
    # Initialize the MediaPipe Hands model for video (different from static image one initialized elsewhere).
    hands_vid = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    frame_with_one = None

    frame_number = 0  # Initialize frame number counter.
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video.

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_vid.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Get the number of detected hands.
            num_hands = len(results.multi_hand_landmarks)
            if num_hands >= n:
                # If n or more hands are detected, return the frame number.
                cap.release()  # Release the video capture object.
                hands_vid.close()  # Close the MediaPipe Hands model.
                return frame_number, n
            elif num_hands == 1 and frame_with_one is None:  # if haven't found one already, identify a frame with just 1 hand
                frame_with_one = frame_number
        frame_number += 1

    cap.release()
    hands_vid.close()
    # If no hands are found in the entire video, return -1.
    if frame_with_one is None:
        return -1, 0
    else:  # return frame with 1 hand, indicate 1 hand detected
        return frame_with_one, 1


def create_zip_file(file_paths: list[str], zip_file_path: str) -> None:
    """
       Creates a zip file containing a specified list of files. The zip file includes a dedicated
       folder for these files and two additional empty directories.

       This function takes a list of file paths and writes them into a zip file, placing the files
       in a subdirectory named 'frames'. Additionally, it creates two empty directories within
       the zip file named 'holding_transitions' and 'not_holding_transitions'.

       Args:
           file_paths (list): A list of strings, where each string is a path to a file that should
                              be included in the zip file. These files will be placed in the 'frames'
                              subdirectory within the zip file.
           zip_file_path (str): The path (including the file name) where the zip file will be created.
                                This is the destination file path for the zip file.

       Returns:
           None: This function does not return any value. It results in the creation of a zip file
                 at the specified location with the specified contents.

       Example:
           >>> file_paths = ['/path/to/frame1.jpg', '/path/to/frame2.jpg']
           >>> create_zip_file(file_paths, 'output/extracted_frames.zip')

       Note:
           - The function will overwrite the zip file at `zip_file_path` if it already exists.
           - The directories 'frames', 'empty folder 1', and 'empty folder 2' are hardcoded and
             will always be created in the zip file. The 'frames' directory will contain all the
             files from the `file_paths` list, while 'empty folder 1' and 'empty folder 2' will be empty.
       """
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # Directory for frames
        frame_dir = "frames"

        # Add empty directories
        for dir_name in [frame_dir, "holding_transitions", "not_holding_transitions"]:
            dir_path = f"{dir_name}/"  # Adding trailing slash to indicate a directory
            zipf.writestr(dir_path, '')  # Add empty directory

        # Add files to the "frames" directory in the zip
        for file in file_paths:
            frame_name = os.path.basename(file)
            frame_path = os.path.join(frame_dir, frame_name)
            zipf.write(file, frame_path)


if 'count' not in st.session_state:
    st.session_state['count'] = 0


def st_increment_counter():
    st.session_state['count'] += 1


# --------------------------------------- SideBar ---------------------------------------
st.sidebar.title("Haptic Categorization Video Preprocessing")
st.sidebar.markdown('''
<H2>This tool can:</H2>

1. Create a cropped video centered on hands
2. Extract frames from video so they can be labeled for AI training

''', unsafe_allow_html=True)
st.sidebar.markdown("<H2>How to Use:</H2>", unsafe_allow_html=True)
st.sidebar.markdown('''
<ol>
    <li>Upload a video</li>
    <li>Hands will be automatically located</li>
    <li>Preview zoom-in centered on hands</li>
    <li>Click "Crop Video" if you want a cropped video or "Extract Frames" if you're labelling for AI training</li>
    <li>Download files when prompted</li>
</ol>
''', unsafe_allow_html=True)
st.sidebar.markdown("<H2>Notes and How it Works:</H2>", unsafe_allow_html=True)
st.sidebar.markdown('''
<ul>
    <li>The software scans each frame in the uploaded video until it finds one where 2 hands can be identified. If there are no such frames, it settles for the first frame where 1 hand is found.</li>
    <li>When uploading a new video, you <em>must</em> click "Reset Session and Hand Detection" to automatically locate hands in the new video</li>
</ul>
''', unsafe_allow_html=True)


st.sidebar.markdown("Created by [**Noah Ripstein**](https://www.noahripstein.com) for the [**Goldreich Lab**](https://pnb.mcmaster.ca/goldreich-lab/CurrentRes.html)")


# --------------------------------------- Main Body ---------------------------------------
uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_video is not None:
    # tempfiles needed for streamlit and opencv to interact properly
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    if st.session_state['count'] == 0:
        with st.spinner("Finding frame with detected hands:"):
            # automatically find frame with 2 hands detected to set up crop
            selected_frame, found_hands = frame_with_detected_hands(tfile.name, 2)
            st.session_state["selected_frame"] = selected_frame
            st.session_state["found_hands"] = found_hands

    # add buttons
    frame_num = st.number_input("Enter Frame With Detected Hands", min_value=1, value=st.session_state["selected_frame"] + 1, step=1)
    flip = st.checkbox("Flip Vertically", on_change=st_increment_counter)

    if st.button("Reset Session and Hand Detection"):
        st.session_state["count"] = 0
        st.rerun()

    frame = extract_frame_n_st(tfile.name, frame_num)
    processed_image, detection_result, output_tuple = detect_hands_and_draw_bbox_st(frame)

    if detection_result.multi_hand_landmarks is None:  # if no hands are detected, display normally
        if flip:
            processed_image = np.flipud(processed_image)  # this needs to be done separately for after bounds
        st.image(processed_image,
                 use_column_width=True, channels="BGR")
        st.warning(f"No hands detected! (manual crop required)")
        st.stop()

    bounds_sized = image_bounds_set_size_method(output_tuple)
    cropped_image = cv2.rectangle(processed_image.copy(),
                                  pt1=(bounds_sized[0][0], bounds_sized[1][0]),
                                  pt2=(bounds_sized[0][1], bounds_sized[1][1]),
                                  color=(255, 0, 255), thickness=2)

    if flip:
        cropped_image = np.flipud(cropped_image)

    # Preview crop
    st.image(cropped_image, caption=f"Crop Preview: {st.session_state['found_hands']} hands detected", use_column_width=True, channels="BGR")

    if st.button("Crop Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_cropped_vid:
            output_path = tmp_cropped_vid.name
        crop_video(tfile.name, output_path, bounds_sized, vert_flip=flip)

        # Provide the download button
        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download Cropped Video",
                data=file,
                file_name=f"{os.path.splitext(uploaded_video.name)[0]}c.mp4",
                mime="video/mp4"
            )

    if st.button("Extract Frames"):
        with tempfile.TemporaryDirectory() as tmp_frame_folder:
            output_path = tmp_frame_folder

            # Call function to process video and extract frames
            generated_files = crop_and_extract(tfile.name, output_path, bounds_sized, vert_flip=flip,
                                               suffix=os.path.splitext(uploaded_video.name)[0])

            if generated_files:
                zip_file_path = f"{os.path.splitext(uploaded_video.name)[0]}_frames.zip"
                create_zip_file(generated_files, zip_file_path)

                with open(zip_file_path, "rb") as file:
                    st.download_button(
                        label="Download Extracted Frames",
                        data=file,
                        file_name=zip_file_path,
                        mime="application/zip"
                    )

