# Kmeans approximation algorithms for GPU usage with pytorch.
---
### Note: This implementation supports multi-GPUs!!!

### Screen recordings for this project.
https://www.youtube.com/playlist?list=PL3obF89OwHs6CEZK9VaItPTAY6AVw_d6s
----

### Dependencies:
```sh
   pip install torch
```

### Current State:
Implementing center initalizers:
- [x] naive kmeans (uniform distribution)
- [x] kmeans++ (D^2 distribution)
- [ ] kmeans-|| Oversampling

## Implementations:
---
1. Lloyd's Algorithm (a.k.a Naive Kmeans)
2. kmeans++
3. kmeans-||
