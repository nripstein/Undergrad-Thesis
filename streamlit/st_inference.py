import os
import cv2
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
import tempfile
import streamlit as st
import ruptures as rpt


# streamlit run streamlit/st_inference.py --server.maxUploadSize 1024 # run this in terminal for local running

st.set_page_config(page_title="Contact Duration")  # Set tab title

# Hide hamburger menu and Streamlit watermark
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --------------------------------------- Functions ---------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_dir: str = os.path.join(os.path.dirname(os.getcwd()), 'models', 'b0_2_epochs_half_data_h5.h5')) -> tf.keras.Model:
    tf_model = tf.keras.models.load_model(model_dir)
    tf_model = tf.keras.models.load_model("models/b0_2_epochs_half_data_h5.h5")
    return tf_model


def preprocess_frames(frames, input_size: tuple[int, int] = (224, 224), to_rgb: bool = True, normalize: bool = False, flip_vertical: bool = False):
    """
    Preprocess a list of frames.

    Args:
        frames (list of numpy arrays): List of frames to be preprocessed.
        input_size (tuple): A tuple specifying the target size for each frame (width, height).
        to_rgb (bool): Whether to convert from BGR to RGB (needed if extracting frames with OpenCV).
        normalize (bool): Whether to normalize (False for EfficientNet models, True otherwise).
        flip_vertical (bool): Whether to flip the image vertically.

    Returns:
        list of numpy arrays: List of preprocessed frames with the following transformations applied:
            1. Resized to the specified input size.
            2. (Optional) Converted from BGR to RGB color format.
            3. (Optional) Normalized.
            4. (Optional) Flipped vertically.
    """
    preprocessed_frames = []

    for frame in frames:
        # Resize the frame to the specified input size
        frame = cv2.resize(frame, input_size)

        # Convert the frame from BGR to RGB color format if needed
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to between 0 and 1 if needed
        if normalize:
            frame = frame / 255.0

        # Flip the image vertically if needed
        if flip_vertical:
            frame = cv2.flip(frame, 0)

        preprocessed_frames.append(frame)

    return np.array(preprocessed_frames)


def predict_frames(tf_model: tf.keras.Model, frames: np.ndarray) -> np.ndarray:
    """
    Predict the class for each frame.

    Parameters:
    - model (tf.Model): Trained TensorFlow model.
    - frames (list): List of frames which are preprocessed from the preprocess_frames() function.
    - input_size (tuple): Tuple indicating the input size (height, width) expected by the model.

    Returns:
    - predictions (np.ndarray): Array of predictions (0s and 1s).
    """
    pred_probs = tf_model.predict(frames)
    return pred_probs.flatten()


def frame_count_approx(video_path: str) -> int:
    """Gets approximate framecount of video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def load_video_in_batches(video_path: str, batch_size: int = 64) -> np.ndarray:
    """
    Load a video and yield batches of frames.

    This function reads frames from a video file and yields them in batches. Each batch contains a specified
    number of frames. The last batch may contain fewer frames if the total number of frames in the video
    is not divisible by the batch size.

    Parameters:
    - video_path (str): Path to the video file.
    - batch_size (int): Number of frames per batch.

    Yields:
    - batch_frames (np.ndarray): Batch of frames as numpy arrays.
    """

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            if len(frames) == batch_size:
                yield np.array(frames)
                frames = []  # Clear the frames for next batch
        else:
            # Yield remaining frames if any
            if frames:
                yield np.array(frames)
            break

    cap.release()


def process_video(video_path: str, tf_model: tf.keras.Model, batch_size: int = 64):
    """
        Process a video file in batches and predict using a TensorFlow model.

        This function divides the video into batches of frames, preprocesses each batch,
        and then uses the TensorFlow model to make predictions. Progress is displayed
        using a Streamlit progress bar.

        Args:
            video_path (str): Path to the video file.
            tf_model (tf.keras.Model): A pre-trained TensorFlow model for making predictions.
            batch_size (int): Number of frames to process in each batch.

        Returns:
            list: A list containing predictions for each batch of frames.
        """
    full_preds = []
    current_frame = 0  # To keep track of the number of frames processed

    progress_bar = st.progress(0)
    for batch_frames in load_video_in_batches(video_path, batch_size):
        preprocessed_frames = preprocess_frames(batch_frames)
        predictions = predict_frames(tf_model, preprocessed_frames)
        full_preds.append(predictions)

        # Update the current frame count and progress bar
        current_frame += len(batch_frames)
        progress_percent = current_frame / total_frames
        progress_bar.progress(progress_percent, f"Classifying frames... {progress_percent * 100:.0f}%")
    progress_bar.progress(100, "Classified frames")
    return np.concatenate(full_preds)


def plotly_changepoint(line_data, changepoints: list[int] = None, fps: float = 29.97, save: bool = False):
    """
    Create an interactive line chart with one or two lines.

    Args:
        line_data (numpy array): First line data (e.g., probability of contact).
        changepoints (list of int): List of indices to add black dotted vertical lines.
        fps (float): Fps for x-axis
        save (bool or str): False, or name of file to save to.

    Returns:
        None
    """
    # Create a Plotly figure
    fig = go.Figure()

    # Add a trace for the first line (e.g., blue pixel counts)
    fig.add_trace(go.Scatter(x=np.arange(len(line_data)) / fps,
                             y=line_data,
                             mode='lines',
                             name='Probability of Contact'))

    # Add vertical lines at changepoints
    if changepoints is not None:
        for index in changepoints:
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=index/fps,
                    x1=index/fps,
                    y0=0,
                    y1=1,
                    line=dict(color="red", dash="dot")
                )
            )

    # Customize the layout
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Value",
        title=f"Probability of Contact"
    )

    # Enable zooming and panning
    fig.update_xaxes(type='linear')
    fig.update_yaxes(type='linear')

    if save:
        fig.write_html(save)

    # Display the interactive plot
    # fig.show()
    st.plotly_chart(fig)
    # return fig


# --------------------------------------- Main Body ---------------------------------------
uploaded_video = st.file_uploader("Upload Video", type=["mp4"])
batch_size = st.number_input("Frame inference batch size (change at own risk)", value=64)

with st.spinner("Loading image classifier"):
    model = load_model("models/b0_2_epochs_half_data_h5.h5")

if uploaded_video is not None:
    # tempfiles needed for streamlit and opencv to interact properly
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    total_frames = frame_count_approx(tfile.name)

    if "predictions" not in st.session_state:
        predictions = process_video(tfile.name, model, batch_size=batch_size)
        st.session_state["predictions"] = predictions
        plotly_changepoint(predictions, fps=1)

    pelt_param = st.number_input("Pelt parameter: (5-10 is reasonable range)", step=1, value=10)

    if st.button("Perform time series segmentation"):
        algo = rpt.Pelt(model="l2", min_size=15).fit(st.session_state["predictions"])
        my_bkps = algo.predict(pen=pelt_param)

        plotly_changepoint(st.session_state["predictions"], my_bkps)
        st.success(f"{len(my_bkps)} breakpoints")






