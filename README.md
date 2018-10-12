# Two-dimensional (2D) PCA & Two-directional two-dimensional (2D2D) PCA 
2D PCA: https://ieeexplore.ieee.org/document/1261097  
2D2D PCA: https://www.sciencedirect.com/science/article/pii/S0925231205001785

実装 -> pca_two_dim.py  
sample(画像再構成) -> image_reconstruction.py  
semple(画像補完) -> image_complement.py

# Sample Run
```
$ ./image_reconstruction.py [複数枚の画像が入ったディレクトリのパス] [再構成を行う画像のパス]
$ ./image_complement.py [複数枚の画像が入ったディレクトリのパス] [再構成を行う画像のパス]
```

## example
```
$ ./image_reconstruction.py ./att_face/s2 ./att_face/s2/1.pgm
```
## data set example
https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html