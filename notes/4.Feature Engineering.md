

# Feature Scaling

* Feature Scaling is an important step to take prior to training of machine learning
models to ensure that features are within the same scale.
* Example: interest rate and employment score are at a different scale. This will result
in one feature dominating the other feature.
* Scikit Learn offers several tools to perform feature scaling.


## Normalization

`Normalization` is conducted to make feature values range from 0 to 1

![[Pasted image 20230531101607.png]]
![[Normalization.png]]

the objective of the `normalization` process is to transform the` min / max` value from what ever number it is to `0 and 1`.


## Standardization

![[Standardization.png]]

![[Pasted image 20230531111527.png]]
![[stand formual.png]]


*"A normalized dataset will always range from 0 to I"*

*"A standardized dataset will always have a mean
of 0 and standard deviation of 1, but can have any
upper and lower values"*

### when should we perform standaridization Vs Normalization


* Scaling (standardization or normalization) is required
when we use any machine learning algorithm that require
gradient calculation.

* Examples of machine learning algorithms that require
gradient calculations are: linear/logistic regression and
artificial neural networks
* Having different scales for each feature will result in a
different step size which in turn jeopardizes the process of
reaching a minimum point.
* Scaling is not required for distance-based and tree-based
algorithms such as `K-Means Clustering,` Support Vector
Machines and K Nearest Neighbors, decision trees,
random forest, and XG-Boost.


![[standaridization Vs Normalization.png]]


**Video Links :**

![standaridization Vs Normalization](https://www.youtube.com/watch?v=bqhQ2LWBheQ)

![Z-Scores, Standardization, and the Standard Normal Distribution](https://www.youtube.com/watch?v=2tuBREK_mgE)


