# Image Classification

Traffic congestion seems to be at an all-time high. Machine Learning methods must be developed to help solve traffic problems. 

Analyze features extracted from traffic images depicting different objects to determine their type as one of 11 classes, noted by integers 1-11: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle. The object classes are heavily imbalanced. For example, the training data contains 10375 cars but only 3 bicycles and 0 people. Classes in the test data are similarly distributed.

The training dataset consists of 21186 records and the test dataset consists of 5296 records.

### Objectives

* Use/implement a feature selection/reduction technique. 
* Experiment with various classification models.
* Deal with imbalanced data.
* Evaluate using F1 Scoring Metric.

### Features in the dataset

- 512 Histogram of Oriented Gradients (HOG) features
- 256 Normalized Color Histogram (Hist) features
- 64 Local Binary Pattern (LBP) features 
- 48 Color gradient (RGB) features
- 7 Depth of Field (DF) features

### Links

* [Report](report/012457289.pdf)
* [Jupyter Notebook](src/image_classifier_pr2.ipynb)

### F1-Score = 0.8502



