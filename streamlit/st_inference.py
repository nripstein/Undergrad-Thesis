import cv2
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
import tempfile
import streamlit as st
import ruptures as rpt
from tqdm import tqdm
from collections.abc import Generator


# streamlit run streamlit/st_inference.py --server.maxUploadSize 1024 # run this in terminal for local running

st.set_page_config(page_title="Segmentation Inference")  # Set tab title

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
def load_model(model_dir: str = None) -> tf.keras.Model:
    tf_model = tf.keras.models.load_model(model_dir) # "models/b0_2_epochs_half_data_h5.h5"
    # tf_model = tf.keras.models.load_model("models/b0_2_epochs_half_data_h5.h5")
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


def load_video_in_batches(video_path: str, batch_size: int = 64) -> Generator[np.ndarray]:
    """
    - Load a video and yield batches of frames.

    Reads frames from a video file and yields them in batches. Each batch contains a specified
    number of frames. The last batch may contain fewer frames if the total number of frames in the video
    is not divisible by the batch size.

    Parameters:\n
    - video_path (str): Path to the video file.
    - batch_size (int): Number of frames per batch.

    Yields:\n
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
    with tqdm(total=total_frames, desc="Classifying frames", ncols=0, ascii=True, position=0) as pbar:
        for batch_frames in load_video_in_batches(video_path, batch_size):
            preprocessed_frames = preprocess_frames(batch_frames)
            predictions = predict_frames(tf_model, preprocessed_frames)
            full_preds.append(predictions)

            # Update the current frame count and progress bar
            current_frame += len(batch_frames)
            pbar.update(len(batch_frames))

            # Estimate remaining duration
            itr_rate = pbar.format_dict["rate"]
            remaining_s = (pbar.total - pbar.n) / itr_rate if itr_rate and pbar.total else 0

            progress_percent = current_frame / total_frames
            progress_bar.progress(progress_percent, f"Classifying frames... {progress_percent * 100:.1f}% {int(remaining_s // 60):02d}:{int(remaining_s % 60):02d} remaining")

    progress_bar.progress(100, "Classified frames")
    return np.concatenate(full_preds)


def plotly_changepoint1(line_data, changepoints: list[int] = None, fps: float = 29.97,) -> go.Figure:
    """
    Create an interactive line chart with one or two lines.

    Args:
        line_data (numpy array): First line data (e.g., probability of contact).
        changepoints (list of int): List of indices to add black dotted vertical lines.
        fps (float): Fps for x-axis

    Returns:
        plotly figure
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
        title="Probability of Contact"
    )

    # Enable zooming and panning
    fig.update_xaxes(type='linear')
    fig.update_yaxes(type='linear')
    return fig


def plotly_changepoint(line_data, changepoints: list[int] = None, fps: float = 29.97, true_changepoints: list[int] = None) -> go.Figure:
    """
    Create an interactive line chart with one or two lines and optional shading between true_changepoints.

    Args:
        line_data (numpy array): First line data (e.g., probability of contact).
        changepoints (list of int): List of indices to add black dotted vertical lines.
        fps (float): Fps for x-axis
        true_changepoints (list of int): List of indices for alternating shading regions.

    Returns:
        plotly figure
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
                    x0=index / fps,
                    x1=index / fps,
                    y0=0,
                    y1=1,
                    line=dict(color="red", dash="dot")
                )
            )

    # Add alternating shading regions based on true_changepoints
    if true_changepoints is not None:
        for i in range(len(true_changepoints) - 1):
            color = "lightblue" if i % 2 == 0 else "red"
            fig.add_shape(
                go.layout.Shape(
                    type="rect",
                    x0=true_changepoints[i] / fps,
                    x1=true_changepoints[i + 1] / fps,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    opacity=0.3,
                    layer="below",
                    line=dict(width=0)
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

    return fig


# --------------------------------------- SideBar ---------------------------------------
st.sidebar.title("Haptic Categorization Video Contact Duration Detection")
st.sidebar.markdown('''
<H2>This tool:</H2>

THIS SECTION ISNT WRITTEN

''', unsafe_allow_html=True)
st.sidebar.markdown("<H2>How to Use:</H2>", unsafe_allow_html=True)
st.sidebar.markdown('''
<ol>
    <li>Upload a video</li>
    <li>When processing is done, click 'perform time series segmentation'"</li>
    <li>More to come (maybe include evaluation of a time series thing based on labeled transition frames)</li>
</ol>
''', unsafe_allow_html=True)
st.sidebar.markdown("<H2>Notes and How it Works:</H2>", unsafe_allow_html=True)
st.sidebar.markdown('''
<ol>
    <li>The tool extracts the number of frames corresponding to the batch size</li>
    <li>Each frame in the batch is assigned a probability that the the participant is holding the object</li>
    <li>Steps 1 and 2 are repeated until every frame of the video has been processed</li>
    <li>The time series is segmented into "contact" and "non-contact" segments using the <a href="https://arxiv.org/abs/1101.1438">PELT algorithm</a></li>
</ol>
''', unsafe_allow_html=True)

st.sidebar.markdown("<H2>Batch Size Parameter:</H2>", unsafe_allow_html=True)
st.sidebar.markdown('''
<ul>
    <li>Batch size is the number of frames stored in memory which are classified at the same time</li>
    <li><em>64 is a reasonable starting point, probably doesn't need to be changed</em></li>
    <li>High batch size requires more memory, optimal batch size depends on computer's memory</li>
    <li>Increasing beyond 64 has marginal benefits in testing</li>
    <li>Decreasing below 32 has noticeable slowdown</li>
    <li>Powers of 2 will likely have slightly better performance</li>
</ul>
''', unsafe_allow_html=True)


st.sidebar.markdown("Created by [**Noah Ripstein**](https://www.noahripstein.com) for the [**Goldreich Lab**](https://pnb.mcmaster.ca/goldreich-lab/CurrentRes.html)")


# --------------------------------------- Main Body ---------------------------------------
uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

if st.button("Reset Session"):
    st.session_state["count"] = 0
    st.rerun()
fps = st.number_input("FPS", value=1)
batch_size = st.number_input("Batch size", value=64)


with st.spinner("Loading image classifier"):
    # model = load_model("models/b0_2_epochs_half_data_h5.h5")
    # model = load_model("models/sr_sv_nr_v2-l-4e-1f.h5")
    model = load_model("models/sr_sv_nr_b0_0.4_3t_1f.h5")

if uploaded_video is not None:
    # tempfiles needed for streamlit and opencv to interact properly
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    total_frames = frame_count_approx(tfile.name)

    if "predictions" not in st.session_state:
        predictions = process_video(tfile.name, model, batch_size=batch_size)
        st.session_state["predictions"] = predictions
        st.plotly_chart(plotly_changepoint(predictions, fps=fps))

    pelt_param = st.number_input("PELT parameter: (5-10 is reasonable range)", step=1, value=10)

    if st.button("Perform time series segmentation"):
        algo = rpt.Pelt(model="l2", min_size=15).fit(st.session_state["predictions"])
        my_bkps = algo.predict(pen=pelt_param)

        st.plotly_chart(plotly_changepoint(st.session_state["predictions"], my_bkps, fps=fps, true_changepoints=[15, 30, 200]))
        st.success(f"{len(my_bkps)} breakpoints")
