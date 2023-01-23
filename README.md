# Data-Mining 
Data Mining Project for the course (Applied Mathematics and Computer Science)

### Description

### Part 1 - Supervised Learning 

The aim of this study is to predict the varieties of dry beans using binary classification. The problem consists of seven
different types of dry beans used in a research, taking into account the variables such as form, shape, type, and structure
by the market situation. For the classification model, images of 13,611 grains of 7 different registered dry beans were
taken with a high-resolution camera. Bean images obtained by computer vision system were subjected to segmentation
and feature extraction stages, and a total of 16 variables; 12 contain information regarding different dimension measures
and the remaining about shape forms, were obtained from the grains. The original data had observations referring to 7
different bean varieties, in this study’s case, only two varieties were considered. Therefore, the objective of this work is
to apply models that correctly determine the variety. The variables associated to the data are:

• Area (A)- The area of a bean zone and the number of pixels within its boundaries.

• Perimeter (P)- Bean circumference is defined as the length of its border.

• Major axis length (L)- The distance between the ends of the longest line that can be drawn from a bean.

• Minor axis length (l)- The longest line that can be drawn from the bean while standing perpendicular to the main
axis.

• Aspect ratio (K)- Defines the relationship between L and l.

• Eccentricity (Ec)- Eccentricity of the ellipse having the same moments as the region.

• Convex area (C)- Number of pixels in the smallest convex polygon that can contain the area of a bean seed.

• Equivalent diameter (Ed)- The diameter of a circle having the same area as a bean seed area.

• Extent (Ex)- The ratio of the pixels in the bounding box to the bean area.

• Solidity (S)- Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.

• Roundness (R)

• Compactness (CO)- Measures the roundness of an object.

• ShapeFactor1 (SF1).

• ShapeFactor2 (SF2).

• ShapeFactor3 (SF3).

• ShapeFactor4 (SF4).

• Response variable: Beans can be one of two varieties, BOMBAY or DERMASON.

The report consists of 6 sections and is organized as follows: Section 2 describes the dataset and some preliminary
analysis, Section 3 describes the methodology, Section 4 describes the classification methods, Section 5 presents the
experimental findings and Section 6 concludes.

### Part 2 - Unsupervised Learning and Supervised Learning on Clustering

This project is a continuation of the first part. The initial stage of the study contains an analysis of the dataset including
two different types of dry beans in order to build an appropriate supervised binary classification algorithm for predicting
dry bean varieties. However, since one has abstracted from the idea of having a labeled dataset and is simply interested
in the feature space, the problem is now viewed from a new perspective. This way, the goal is to see if there are any
interesting patterns or relationships between the bean observations using clustering algorithms, which will lead to the
conclusion that some observations are related in some way and hence belong together. Cluster Analysis is based on
the idea of identifying groups (or clusters) of items that are as similar as possible while differing from observations from
other groups.
To validate the resulting clusters, external assessment indices (Accuracy, Sensitivity, Specificity, Balanced Accuracy,
Precision, F1-score) and the confusion matrices are used to quantify and validate the decisions taken. It is worth noting
that one will be using the real beans classes to subsequently validate the discovered clusters.
The next step is to apply the two best classifiers from Part 1 to this data, treating the best clustering solution as the
new "classes" of the response variable, and compare the classes predicted by classification algorithms when using the
clustering data and the real data. Thus, in the first part of this project, a supervised learning study was performed, from
which it was concluded that the classifiers that best fit the proposed dataset are: KNN and XGBoost. In this second part,
an unsupervised evaluation of the problem will be performed by applying the following clustering methods: Hierarchical
(Single Linkage, Complete Linkage, Average Linkage and Ward’s Method), Partitioning (K-Means, K-Medoids), Density
Based (DBSCAN), Graph Based (Mst-Knn), Spectral.
In this manner, one chooses a strategy to achieve objectives of the project. The datasets that will be used to run the
clustering methods are described in section 2. Section 3 introduces the technique used for determining the appropriate
clustering algorithm as well as the dataset that will be utilized for classification. In section 4, the approach for comparing
the results produced from classifiers trained on real classes to classifiers trained on clustered classes is proposed. By
section 5 one begin to describe each clustering method used and some of the results obtained, in section 6, one provides
the final results of methods to answer our problem and and in the last section one can find the conclusions.
