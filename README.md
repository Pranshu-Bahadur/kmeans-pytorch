# Kmeans approximation algorithms for GPU usage with pytorch.
---
### Note: This implementation supports multi-GPUs!!!

### Dependencies:
```sh
   pip install torch torchvision numpy
```

## Implementations:
---
1. Lloyd's Algorithm (a.k.a Naive Kmeans)
2. kmeans++
3. kmeans-||


## Notes:

Okay...What about the structure?

It shouldn't confuse the user.

kmeans.py

Yeah I don't really see the point of multiple files.

Okay that's kinda dumb. Code should be understandable.

Idk making it an object makes no sense. Fuck it lets brute force this.

psuedocode:
function clustering(x, k, seeder, **kwargs):
        centers <- seeder(x, k, **kwargs['seeder_params']) # single dispatch
        states = {inf:()}
        while True:  
                curr_cost, state  <- compute(x, centers, **kwargs['compute_params'])
                if min(states.keys()) is curr_cost:
                        return curr_cost, state[curr_cost]
                states[curr_cost] := state


        1. centers
        2. distances
        3. clusters acc to argmin
        4. cost





