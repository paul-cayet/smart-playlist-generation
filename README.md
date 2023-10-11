![smart playlist generator title](docs/smart_playlist_header.jpg)


# Smart playlist generator


This Python tool is designed to generate satisfying playlists from a selection of music files.


![smart playlist generator figure](docs/smart_playlist_img.jpg)


The tools works in 4 simple steps:

1. The `FeatureGenerator` computes vector representations of music segments, which are then processed if necessary by a `FeatureProcessor` (e.g. averaging the features)
2. A cost matrix $C$ of the best transitions timestamps between all pairs of music: given $n$ music files, we compute $C$ such that
$$\forall i \neq j \in [1,n], C_{i,j}= ||Z_i(t_i^*)-Z_j(t_j^*)|| $$

with
  -  $Z$ being the music representation
  - $t_i^*$ being the optimal time to transition in/out of music $i$

3. We optimize the playlist order using the cost matrix $C$ defined above (currently the problem is modeled as a Travelling Salesman Problem, and in our case the looping condition is optional)
4. We use the solution to generate the final playlist and apply linear/sigmoid fading to form the transitions


## Technology

#### Feature Representations

I read a few paper to figure out the best way to build music feature representation.

- M. C. McCallum, “Unsupervised learning of deep features for music segmentation,” in ICASSP 2019, pp.346–350.
- P. G. Antunes, “Audio-based Music Segmentation Using Multiple Features”
- J. Salamon, O. Nieto, N. J. Bryan, “Deep Embeddings and Section Fusion Improve Music Segmentation”, in Proc. of the 22nd Int.
Society for Music Information Retrieval Conf., Online, 2021

While the Deep Learning approach looked promising, it required substantial amounts of data and was not appropriate as a initial solution. Instead, I decided to rely on the widely used Mel Frequency Cepstral Coefficients (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), that are supposed to be aligned with the way human auditory system's perceives audio.


#### Optimization problem

We model the playlist ordering problem as a Travelling Salesman Problem, a well-known NP-Hard combinatorics problem.

Finding the exact solution takes a $O(n!)$ time so we rely on approximate solutions such as metaheuristics. Here we use the Simulated Annealing algorithm.

For more information, read my blog posts on solving the TSP with metaheuristics ([part1](https://paul-cayet.github.io/2022/05/15/sa_article1.html), [part 2](https://paul-cayet.github.io/2022/05/18/sa_article2.html))


## Using the tool

Generating an optimal playlist only takes a few steps.

1. Install the library

```
./setup.sh
```
which is equivalent to

```
python3 -m venv .venv
source ".venv/bin/activate"

pip install -r requirements.txt
pip install .
```

Then use the tool with the appropriate parameters:

```
python src/smart_playlist_geneation/main.py \
    --music_folderpath "..." \
    --saving_folderpath "..." \
```
