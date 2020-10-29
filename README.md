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
- The matrix is a $\mathbb{N}^6 \times \mathbb{N}^6$. A square matrix of 6 interval vectors is produces for each Analysis start point.

#### Capturing Dynamics

We also wanted to capture the dynamics of a live performance in a SSM in parallel with the interval vectors. What we did is for the same windows as before we create matrices of size $128 \times 6$. For each Midi note $[0, 127]$ we add its midi velocity value inside the window. Therefore we obtain non-negative submatrices of the same size.
    
    
    We this method we capture the dynamic tension of segments but as well the appearance of notes in a segment.


### Towards Segmentation Techniques with Learning

In this section we discuss techniques and give definition to problems regarding the segmentation of the obtained SSMs.

#### Formal Definition of Paths and Blocks

We formally define a **segment** to be a set $\alpha=[s:t]\subseteq [1:N]$ specified by its starting point $s$ and its end point $t$ (given in terms of feature indices). Let 

\begin{equation}
    |\alpha|:=t-s+1
\end{equation}

denote the length of $\alpha$. Next, a **path** over $\alpha$ of length $L$ is a sequence

\begin{equation}
   P=((n_1,m_1), \ldots,(n_L,m_L))
\end{equation}   

of cells $(n_\ell,m_\ell)\in[1:N]^2$, $\ell\in[1:L]$, satisfying $m_1=s$ and $m_L=t$ (**boundary condition**) and $(n_{\ell+1},m_{\ell+1}) -(n_\ell,m_\ell)\in \Sigma$ (**step size condition**), where $\Sigma$ denotes a set of admissible step sizes. Note that this definition is very similar to the one of a [warping path](../C3/C3S2_DTWbasic.html) (see Section 3.2.1.1 of <a href="http://www.music-processing.de/">[Müller, FMP, Springer 2015]</a>). In the case of $\Sigma = \{(1,1)\}$, one obtains paths that are strictly diagonal. In the following, we typically use the set $\Sigma = \{(2,1),(1,2),(1,1)\}$. For a path $P$, one can associate two segments defined by the projections $\pi_1(P):=[n_1:n_L]$ and $\pi_2(P):=[m_1:m_L]$, respectively. The boundary condition enforces $\pi_2(P)=\alpha$. The other segment $\pi_1(P)$ is referred to as the **induced segment**. The **score** $\sigma(P)$ of $P$ is defined as

\begin{equation}
\sigma(P) := \sum_{\ell=1}^L \mathbf{S}(n_\ell,m_\ell).
\end{equation}

Note that each path over the segment $\alpha$ encodes a relation between $\alpha$ and an induced segment, where the score $\sigma(P)$ yields a quality measure for this relation. For blocks, we also introduce corresponding notions. 
A **block** over a segment $\alpha=[s:t]$ is a subset 

\begin{equation}
   B=\alpha' \times \alpha \subseteq [1:N]\times [1:N]
\end{equation}

for some segment $\alpha'=[s':t']$. Similar as for a path, we define the two projections $\pi_1(B)=\alpha'$ and $\pi_2(B)=\alpha$ for the block $B$ and call $\alpha'$ the **induced segment**. Furthermore, we define the score of block $B$ by

\begin{equation}
\sigma(B)=\sum_{(n,m)\in B}\mathbf{S}(n,m).
\end{equation}


Based on paths and blocks, one can consider different kinds of similarity relations between segments. We say that a segment $\alpha_1$ is **path-similar** to a segment $\alpha_2$, if there is a path $P$ of high score with $\pi_1(P)=\alpha_1$ and  $\pi_2(P)=\alpha_2$. Similarly, $\alpha_1$ is **block-similar** to $\alpha_2$, if there is a block $B$ of high score with $\pi_1(B)=\alpha_1$ and $\pi_2(B)=\alpha_2$. Obviously, in case that the similarity measure $s$ is symmetric, both the self-similarity matrix $\mathbf{S}$ and the above-defined similarity relations between segments are symmetric as well. Another important property of a similarity relation is **transitivity**, i.e., if a segment $\alpha_1$ is similar to a segment $\alpha_2$ and segment $\alpha_2$ is similar to a segment $\alpha_3$, then $\alpha_1$ should also be similar to $\alpha_3$ (at least to a certain degree). Also this property holds for path- and block-similarity in case that the similarity measure $s$ has this property. As a consequence, path and block structures often appear in groups that fulfill certain symmetry and transitivity properties&mdash;at least in the ideal case. 