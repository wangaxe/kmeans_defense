# kmeans_defense
**This is code for our paper: Harden Deep Convolutional Classifiers via K-means Reconstruction**

**Requirements**: scikit-learn, pytorch, torchattacks, and advertorch.

Those pre-trained models that are utilized in this work are provided in `/models/`, and the corresponding code for training models can be found in `/Train/` folder. The `/Evaluation/` folder contains the code to reproduce our experiments, and the reported results in our paper (see `/Evaluation/mnist` and `/Evaluation/svhn`). Usage of these codes are shown in `run.sh` and `svhn_run.sh`.