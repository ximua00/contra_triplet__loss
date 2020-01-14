## Pytorch Contrastive and Triplet Loss experiments

### Setup
```conda install --file requirements.txt```

### Run experiments
```python main.py```

### Results
Mean Average Precision@100
| Dataset        | Contrastive Loss  | Triplet Loss | Batch Hard |
| ------------- |:-------------:|:-----:|:-----:|
| MNIST      | 0.986 | 0.983 | -- |
| FashionMNIST | 0.86 | 0.871 | -- |
| CIFAR10 | 0.697 | 0.639 | -- |
| Cars3D | 0.501 | 0.532 | 0.667 |
| CarsEPFL | 0.832 | 0.769 | 0.761 |
| CarsShapeNet | 0.56 | 0.679 | 0.739 |

### Loss Implementations
1. Contrastive Loss
2. Vanilla Triplet loss
3. Batch Hard Triplet Loss
4. Batch Soft Triplet loss

### DataLoaders
1. MNIST
2. FashionMNIST
3. CIFAR10
4. Cars3D
5. CarsEPFL
6. CarsShapeNet

### References
1. [Github Adambielski's siamese-triplet](https://github.com/adambielski/siamese-triplet)
2. [Github Beyond-Binary-Supervision-CVPR19](https://github.com/tjddus9597/Beyond-Binary-Supervision-CVPR19/blob/master/code/Dense_TripletLoss.py)
3. [Github kilsenp's triplet-reid-pytorch](https://github.com/kilsenp/triplet-reid-pytorch/blob/master/triplet_loss.py)
4. [Data Cars3D](https://github.com/carpedm20/visual-analogy-tensorflow/blob/master/download.sh)
5. [Data CarsEPFL](https://www.epfl.ch/labs/cvlab/data/data-pose-index-php/)
6. [Data CarsShapeNet](https://www.shapenet.org)
7. [Paper FaceNet](https://arxiv.org/abs/1503.03832)
8. [Paper In Defense of Triplet Loss](https://arxiv.org/abs/1703.07737)

### TODO
1. Argparser