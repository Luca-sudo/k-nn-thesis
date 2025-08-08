# Hypotheses

- [x] Hypothesis 1: HNSW remains more precise than LSH on clustered data as the number of sites grows.
- [x] Hypothesis 2: Given two clusters in the upper-right quadrant of cartesian space, HNSW's quality remains steady and LSH's quality improves as the distance between the two clusters grows. The distance between clusters directly corresponds to the spread of the underlying distribution -- some pairs of points remain infinitesimally close to each other, while the distance between points of separate clusters is directly affected by the distance between the two clusters.
- [x] Hypothesis 3: On a uniform grid (all sites have integer coordinates within a bounded region), HNSW retains quality whereas LSH degrades in quality as the size of the region is increased.
- [x] Hypothesis 4: The observed loss of quality in hypothesis 3 can be counteracted by increasing the number of separating hyperplanes.
- [x] Hypothesis 5: Over a uniform-grid similar to hypotheses 3 & 4, the quality of LSH diminishes as the dimensionality of the feature space increases over a bounded region.
- [x] Hypothesis 6: Quality of LSH queries degenerates as density of sites increases.
- [ ] Hypothesis 7: Version of hypothesis 3 that translates grid by irrational number to avoid cosine similarity interference.
- [X] Hypothesis 8: Compare LSH & HNSW on Hypersphere. Should be LSH friendly by avoiding colinearity issues.
- [ ] Hypothesis ?: Compare Ball-Tree and k-d Tree 

# TODO
- [X] Add ranks to LSH vs. HNSW plot
- [X] Add recall to LSH vs. HNSW plot
- [ ] Evaluate LSH and HNSW on practical datasets.


