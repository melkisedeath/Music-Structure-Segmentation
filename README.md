# Music-Structure-Segmentation
A repository for Symbolic Music Structural Segmentation with Self-Similarity Matrices


## Overview
In this repository we investigate several techniques for structural Segmentation of symbolic music using Self-Similarity Matrices (SSM). We annotate manually the structure of works by Beethoven, Mozart and Chopin. Finally we will perform Segmentation using Learning techniques.



### Structural Segmentation with Convolutional Techniques


#### Structural Segmentation with submatrices of a constant Size

In this notebook we use two main techniques that use interval Vectors and Dynamics
The idea here is to take the interval vector of a series of notes, and move based on a certain time interval just like doing STFT. We are based on a score informed minimal and median beat. 
The median beat or measure is the analysis size and the minimal beat will be the step size. 
So we take the interval vector of the selection.

Then we perform some matrix Factorisation Technique such as PCA, NMF or NTD.

Our solution consists of using a time window, a step or overlap factor for which we calculate the interval vector move and overlap along a piece or segment of music.
    
Moreover we are considering the use of square matrices and therefore we multiply the window by 6 interval numbers and obtain 6 interval vectors for each start time.
- A step is calculated of being either the minimum beat or time interval between note onsets, or the mean interval between note onsets.
- Each window has the same start but 6 different sizes which are integer multiples of the step size. The next window is moved by a time interval equal to the step size.
- The matrix is a <img src="https://render.githubusercontent.com/render/math?math=\mathbb{N}^6 \times \mathbb{N}^6">. A square matrix of 6 interval vectors is produces for each Analysis start point.

#### Capturing Dynamics

We also wanted to capture the dynamics of a live performance in a SSM in parallel with the interval vectors. What we did is for the same windows as before we create matrices of size <img src="https://render.githubusercontent.com/render/math?math=128 \times 6">. For each Midi note [0, 127] we add its midi velocity value inside the window. Therefore we obtain non-negative submatrices of the same size.
    
    
We this method we capture the dynamic tension of segments but as well the appearance of notes in a segment.


### Towards Segmentation Techniques with Learning

In this section we discuss techniques and give definition to problems regarding the segmentation of the obtained SSMs.

