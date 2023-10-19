<H1>AI Computer Vision Software for Undergrad Thesis</H1>
This repository contains code for my Undergraduate Senior Honours Thesis in the <a href="https://pnb.mcmaster.ca/goldreich-lab/CurrentRes.html#Modeling">Goldreich Lab</a>. I'm developing a computer vision program which can automatically detect how long participants are holding an object in videos. 


<div align="center">
  <H3>Proof of Concept Video</H3>
  <img src="https://github.com/nripstein/Undergrad-Thesis/assets/98430636/57cd803a-3eb8-490a-bf0a-eccc0afdff79" alt="36_prob_gif">
  <p>AI displays probability that participant is touching grey object <em>(AI was not trained on this clip)</em></p>
</div>

<div align="center">
  <H3>Proof of Concept Multi-Trial Labelling</H3>
  <img src="https://github.com/nripstein/Undergrad-Thesis/assets/98430636/e86df995-8077-45a6-8fb8-ea3ff3b7cd91" alt="36_prob_gif">
  <p>Predicted contact duration using simple threshold decision rule. Each shaded region represents duration of a different trial <em>(AI was not trained on this portion of video)</em></p>
</div>

<H1>Repository Table of Contents</H1>

<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/f4da48ca9171c976b927834e5edd429a2aac971c/vid_resize_experiments.ipynb">Auto-Zoom Jupyter Notebook</a></h2>
<ul>
  <li>Series of functions to automatically zoom videos to the desired size, with the participant's hands in the center</li>
  <li>Used MediaPipe for hand detection</li>
  <li>If both the participant's and experimenter's hands are in the frame, focuses on 2 most likely hands (which will be the participants, because the model is trained on ungolved hands.</li>
  <li>If desired crop region outside of photo, white pixels get added to retain desired output video dimensions</li>
  <li>TODO: likely done</li>
</ul>

<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/f4da48ca9171c976b927834e5edd429a2aac971c/vid_resize_experiments.ipynb">Auto-Split Jupyter Notebook</a></h2>
<ul>
  <li>Series of functions to split long videos of many trials into many videos each containing only one trial</li>
  <li>Determines when to split videos based on the amount of blue in a given frame (a trial ends when an experimenter, who wears a blue glove, replaces the object for the participant to classify)</li>
  <li>TODO: likely done</li>
</ul>
<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/922036882ffbd9af599c081a54eedfc0f20cdcb4/vid_classifier_exp.ipynb">Video with Frame Probability Jupyter Notebook</a></h2>
<ul>
  <li>Functions to make predictions in a video using a pre-trained model</li>
  <li>Can save video with probability of contact on each frame overlaid</li>
  <li>Has interactive plotly figure to predict contact time and examine errors</li>
  <li>TODO: need functionality to auto-extract duration</li>
  <ul>
      <li>Need to decide if I'll be using on full videos or clips of one trial</li>
    <li>Need to figure out how to exclude obvious outliers, maybe <a href="https://en.wikipedia.org/wiki/Hysteresis">Hysteresis</a> or anomoly detection of some sort</li>
    </ul>
</ul>
<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/629c88d6e1af0eada75d89550a93fd6de80fea4b/frame_label.ipynb">Frame Label Tool Jupyter Notebook</a></h2>
<ul>
  <li>Easy function to extract frames from video</li>
  <li>Given transition frames (i.e. the first frame of object contact or object non-contact), can automatically label all remaining frames in video</li>
  <li>Saves tons of time labelling: labeller needs to carefully label 10 frames, and it sorts out the remaining 4000+ with perfect accuracy (accuracy guaranteed because it's rule-based, not machine learning-based)</li>
  <li>TODO:</li>
  <ul>
  <li>Add support for edge cases: First or last frame are transition frames</li>
  <li>More functionality for automatically labelling many videos at once</li>
  </ul>
</ul>
<h2><a href="https://github.com/nripstein/Undergrad-Thesis/blob/629c88d6e1af0eada75d89550a93fd6de80fea4b/preprocess_pipeline.py">Preprocess Pipeline</a></h2>
<ul>
  <li>.py file to automatically preprocess videos including auto-zoom and auto-crop (as developed in the Jupyter Notebooks)</li>
  <li>TODO: likely done</li>
</ul>

<H1>Future directions</H1>
<ol>
<li>Need to perform time series analysis for changepoint detection. Will likely detect changepoint if p(touching) makes a huge jump. </li>
</ol>

<H1>Possible Future directions</H1>
<ul>
  <li>MAYBE: Add model which detects clearly visible object with no contact. Then can do similar time series analysis on current type of model (which detects probability of contact in a given frame) and then do ensemble learning. Ideally each model would have good reliability, but together would be very reliable.</li>
  <ul><li>Could do this with another image classification model or maybe with image segmentation</li></ul>
  <li>MAYBE: develop some sort of custom loss function for video duration prediction so I can make a very custom time series segmentation model, probably with a neural network. No real details worked out at this point.</li>
  <ul>
  <li>Could run into same dual training set problem as in Hidden Markov model</li>
  <li>This would probably be extremely interesting, but might constitute graduate-level machine learning research if done properly and would be too much on top of the other aspects of my thesis.</li>
  </ul>
  </ul>
