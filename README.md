# Fivefine

Source code for paper "From Ultra-fine to Fine: Fine-tuning Ultra-fine Entity Typing Models to Fine-grained"

### Prepare

1. In config.py, UF_DIR is where you put the ultra-fine entity typing data.
2. In config.py, WORK_DIR is the work directory.

### UFET Training

1. Train with weak data (A pretrained version will be uploaded later):
   ```python trianuf.py 0```
2. Fine-tune with manual annotation: 
   ```python trianuf.py 1```
3. Self-training: 
   ```python trianuf.py 2```

### FET Training

1. Few-NERD: 
   ```python trainfet.py 0```

### Pre-trained Model

https://drive.google.com/file/d/1z-D1SBvVAF4TFyx_FLuhmqHlqlD2ZN7I/view?usp=sharing

### Other Details

For the BBN dataset, we use the version processed by Ren et al. (2016), but we further split its training set into train/dev.
