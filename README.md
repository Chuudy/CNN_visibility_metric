# CNN visibility metric
> CNN-baded visibility metric which estimate visibile differences between two images.
> Please find more information here: http://visibility-metrics.mpi-inf.mpg.de/

## Requirements

The metric was implemented using python (v 3.5.2) and Tensorflow (v 1.4.0) library.
Additionally, numpy and openCV libraries need to be installed.

## Preapring metric

To run the metric, please proceed with following steps:
1. Clone or download and unpack the repository.
2. Download the network blob from: http://visibility-metrics.mpi-inf.mpg.de/files/NetworkModel.zip.
3. Unpack NetworkModel.zip file to repository root directory.
4. Done! The directory structure should look like this:

```
CNN_visibility_metric
├── .git
├── cnn_visibility.py
└── NetworkModel
    ├── checkpoint
    ├── model.ckpt-2400000.data-00000-of-00001
    ├── model.ckpt-2400000.index
    └── model.ckpt-2400000.meta
```

## Interface

Only a python interface of our metric is provided. The interface takes reference and distorted images as inputs and returns a
visibility map. The visibility map values correspond to probability of detection by population, where higher value means the more likly detection.

The interface supports both, PNG and JPG, images as inputs. To get the visibility map, run:

```
python cnn_visibility.py -r referenceImage.{png|jpg} -t distortedImage.{png|jpg}
```

It is also possible to run metric using more files at the same time. If the corresponding reference and distorted images have the same names and are located in two sepatare directories run:

```
python cnn_visibility.py -r referenceDirectory/* -t distortedDirectory/*
```

The metric will autmatically pair images and process all of them. It is highly recommended to process more than one image same time, since libraries import and network loading processes take significant amount of time.

By default, the results will be saved in 'vismaps' directory, which is created automatically and located in metric root directory.

To direct the output to the custom directory use the third argument as shown below:

```
python cnn_visibility.py -r referenceDirectory/* -t distortedDirectory/* -d customDirectoryName
```

@article{10.1145/3196493,
author = {Wolski, Krzysztof and Giunchi, Daniele and Ye, Nanyang and Didyk, Piotr and Myszkowski, Karol and Mantiuk, Rados\l{}aw and Seidel, Hans-Peter and Steed, Anthony and Mantiuk, Rafa\l{} K.},
title = {Dataset and Metrics for Predicting Local Visible Differences},
year = {2018},
issue_date = {October 2018},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {37},
number = {5},
issn = {0730-0301},
url = {https://doi.org/10.1145/3196493},
doi = {10.1145/3196493},
abstract = {A large number of imaging and computer graphics applications require localized information on the visibility of image distortions. Existing image quality metrics are not suitable for this task as they provide a single quality value per image. Existing visibility metrics produce visual difference maps, and are specifically designed for detecting just noticeable distortions but their predictions are often inaccurate. In this work, we argue that the key reason for this problem is the lack of large image collections with a good coverage of possible distortions that occur in different applications. To address the problem, we collect an extensive dataset of reference and distorted image pairs together with user markings indicating whether distortions are visible or not. We propose a statistical model that is designed for the meaningful interpretation of such data, which is affected by visual search and imprecision of manual marking. We use our dataset for training existing metrics and we demonstrate that their performance significantly improves. We show that our dataset with the proposed statistical model can be used to train a new CNN-based metric, which outperforms the existing solutions. We demonstrate the utility of such a metric in visually lossless JPEG compression, super-resolution and watermarking.},
journal = {ACM Trans. Graph.},
month = nov,
articleno = {172},
numpages = {14},
keywords = {Visual perception, convolutional neural network, data-driven metric, dataset, distortion visibility, image quality, visual difference predictor, visual metric}
}
