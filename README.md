# Undergrad-Thesis
Code for Undergraduate Senior Honours Thesis


<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/f4da48ca9171c976b927834e5edd429a2aac971c/vid_resize_experiments.ipynb">Auto-Zoom Jupyter Notebook</a></h2>
<ul>
  <li>Series of functions to automatically zoom videos to the desired size, with the participant's hands in the center</li>
  <li>Used MediaPipe for hand detection</li>
  <li>If both the participant's and experimenter's hands are in the frame, focuses on 2 most likely hands (which will be the participants, because the model is trained on ungolved hands.</li>
  <li>If desired crop region outside of photo, white pixels get added to retain desired output video dimensions</li>
</ul>

<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/f4da48ca9171c976b927834e5edd429a2aac971c/vid_resize_experiments.ipynb">Auto-Split Jupyter Notebook</a></h2>
<ul>
  <li>Series of functions to split long videos of many trials into many videos each containing only one trial</li>
  <li>Determines when to split videos based on the amount of blue in a given frame (a trial ends when an experimenter, who wears a blue glove, replaces the object for the participant to classify)</li>
</ul>
<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/922036882ffbd9af599c081a54eedfc0f20cdcb4/vid_classifier_exp.ipynb">Video with Frame Probability Jupyter Notebook</a></h2>
<ul>
  <li>Functions to make predictions in a video using a pre-trained model</li>
  <li>Can save video with probability of contact on each frame overlaid</li>
  <li>Has interactive plotly figure to predict contact time and examine errors</li>
</ul>
