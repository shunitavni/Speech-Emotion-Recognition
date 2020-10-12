# Speech-Emotion-Recognition

This is my final project part of my Deep Learning specializing.

The goal of this project was to create an emotion recognition application based on models from the world of deep learning.
The user enters as an input an audio segment in which he speaks and the machine knows how to classify the emotion expressed in the audio segment.

We did in-depth research both around working with audio and around the topic of emotion.<br/>

Emotions are biological states associated with the nervous system brought on by neurophysiological changes variously associated with thoughts, feelings, behavioural responses, and a degree of pleasure or displeasure.
There is currently no scientific consensus on a definition.
Emotions are often intertwined with mood, temperament, personality, disposition, creativity and motivation.

In our work, we concentrated on identifying what is known in the professional literature as Expression.<br/>

**Expression:** facial and **vocal expression** almost always accompanies an emotional state to communicate reaction and intention of actions.<br/>
facial and vocal expression almost always accompanies an emotional state to communicate reaction and intention of actions.<br/>
**With focus on vocal expression of curse**.<br/>

So we can say that emotion is an abstract thing, and so we found it appropriate to classify emotion with the tone of a sentence, that is, to refer to the intonation in which things were said.
In fact with the help of the tone reference we can make a better generalization of our model and classify emotion regardless of what language the things were said in.



_____________________________________________________________________________________________________________________________________

***Data-Set:***

The data-set with which we trained our model,
Contains sentences and each sentence is attached to a label of what
the emotion of the person saying the sentence is. <br/>

Possible Labels:

1. happy
2. sad
3. anger
4. neutral
5. fear
6. disgust


In our quest to classify audio clips, we realized that one set of data would not represent
the distribution of emotion through real-world sound well enough, so throughout the process,
we created new Dataset which contains 4 different Datasets of .wav audio files listed below.

* Surrey Audio-Visual Expressed Emotion (SAVEE)
* Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
* Toronto emotional speech set (TESS)
* Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D)

Total amount of samples in our set (before augmentation): 7,464 <br/>
Total amount of samples in our set (after augmentation): 14,928

_____________________________________________________________________________________________________________________________________

***Data-Set:***

_____________________________________________________________________________________________________________________________________


The code shown is the code of the final application with the specific model we chose to put behind it which ended up classifying only 4 classes.
In the "Architectures" folder you can find all models and experiments that we did during all our work on the project.

The models were written in Python,
The app using FLASK technology (And HTML, CSS, Bootstrap design).
