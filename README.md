
---
## Generalized implementation of kmeans approximation algorithms.

1. Lloyd's Algorithm (naive kmeans) [IMPLEMENTED, UNIT TESTED]
2. kmeans++

### Dependencies

```shell
pip install torch
```
## Examples:
---
```python
from kmeans import NaiveKmeans
x = torch.randn(256, 3)
NaiveKmeans(3)(x)
```

---
To-dos
- [ ] Add a brief description.
- [ ] Make this repo a python package.
- [ ] Add a cool image custom/non-copyrighted image logo for this library.
- [ ] GPU support with dataparrallel
