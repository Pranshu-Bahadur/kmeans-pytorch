# Kmeans approximation algorithms for GPU usage with pytorch.
---
### Note: This implementation supports multi-GPUs!!!

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
