               Bilinear CNN (B-CNN) for Fine-grained recognition


DESCRIPTIONS
    After getting the deep descriptors of an image, bilinear pooling computes
    the sum of the outer product of those deep descriptors. Bilinear pooling
    captures all pairwise descriptor interactions, i.e., interactions of
    different part, in a translational invariant manner.

    B-CNN provides richer representations than linear models, and B-CNN achieves
    better performance than part-based fine-grained models with no need for
    further part annotation.


REFERENCE
    T.-Y. Lin, A. RoyChowdhury, and S. Maji. Bilinear CNN models for
    fine-grained visual recognition. In Proceedings of the IEEE International
    Conference on Computer Vision, pages 1449--1457, 2015.


PREREQUIREMENTS
    Python3.6 with Numpy supported
    PyTorch


LAYOUT
    ./data/                 # Datasets
    ./doc/                  # Automatically generated documents
    ./model/                # Saved models
    ./src/                  # Source code


USAGE
<<<<<<< HEAD
    Step 1. Fine-tune the fc layer only. It gives 77.32% test set accuracy on CUB200 and 70.01% test set accuracy on FGVC-Aircraft.
    $ CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 \
          -H localhost:2 \
          -bind-to none -map-by slot \
          -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
          -mca pml ob1 -mca btl ^openib \
          python ./src/train.py fc --base_lr 1.0 \
          --batch_size 16 --epochs 55 --weight_decay 1e-8 \
          --dataset cub200 \
          | tee "[fc-] base_lr_1.0-weight_decay_1e-8-epoch_.log"

    Step 2. Fine-tune all layers. It gives 83.05% test set accuracy on CUB200 and 82.30% test set accuracy on FGVC-Aircraft.
    $ CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 \
          -H localhost:2 \
          -bind-to none -map-by slot \
          -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
          -mca pml ob1 -mca btl ^openib \
          python ./src/train.py all --base_lr 1e-3 \
          --batch_size 8 --epochs 40 --weight_decay 1e-5 \
          --dataset cub200 --model "model.pth" \
          | tee "[all-] base_lr_1e-2-weight_decay_1e-5-epoch_.log"


LICENSE
    CC BY-SA 3.0
