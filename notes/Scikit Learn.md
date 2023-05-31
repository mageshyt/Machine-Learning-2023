`Refresher` : what is Machine Learning
normally we developer write `function` and give `inputs` to the machine that perform the `task` and give `output`

in ML we give `set of inputs` and Machine Write the `function / Model`

# What we're covering in the Scikit-Learn Introduction

his notebook outlines the content convered in the Scikit-Learn Introduction.

It's a quick stop to see all the Scikit-Learn functions and modules for each section outlined.

What we're covering follows the following diagram detailing a Scikit-Learn workflow.

![[Pasted image 20230519075346.png]]

# Where can I get help?

If you get stuck or think of something you'd like to do which this notebook doesn't cover, don't fear!

The recommended steps you take are:

1.  **Try it** - Since Scikit-Learn has been designed with usability in mind, your first step should be to use what you know and try figure out the answer to your own question (getting it wrong is part of the process). If in doubt, run your code.
2.  **Press SHIFT+TAB** - See you can the docstring of a function (information on what the function does) by pressing **SHIFT + TAB** inside it. Doing this is a good habit to develop. It'll improve your research skills and give you a better understanding of the library.
3.  **Search for it** - If trying it on your own doesn't work, since someone else has probably tried to do something similar, try searching for your problem. You'll likely end up in 1 of 2 places:
    - [Scikit-Learn documentation/user guide](https://scikit-learn.org/stable/user_guide.html) - the most extensive resource you'll find for Scikit-Learn information.
    - [Stack Overflow](https://stackoverflow.com/) - this is the developers Q&A hub, it's full of questions and answers of different problems across a wide range of software development topics and chances are, there's one related to your problem.

An example of searching for a Scikit-Learn solution might be:

> "how to tune the hyperparameters of a sklearn model"

Searching this on Google leads to the Scikit-Learn documentation for the `GridSearchCV` function: [http://scikit-learn.org/stable/modules/grid_search.html](http://scikit-learn.org/stable/modules/grid_search.html)

The next steps here are to read through the documentation, check the examples and see if they line up to the problem you're trying to solve. If they do, **rewrite the code** to suit your needs, run it, and see what the outcomes are.

4.  **Ask for help** - If you've been through the above 3 steps and you're still stuck, you might want to ask your question on [Stack Overflow](https://www.stackoverflow.com/). Be as specific as possible and provide details on what you've tried.

Remember, you don't have to learn all of the functions off by heart to begin with.

What's most important is continually asking yourself, "what am I trying to do with the data?".

Start by answering that question and then practicing finding the code which does it.

Let's get started.

# Choosing the right Estimator/ algorithm for your problem

some things to note:

- Sklearn refer to machine learning models, algorithm as estimators
- classification problem - predicting a category (heart disease or not)
  - Something you'will see `clf` (classifier) used as a classification estimator
- Regression problem - predicting a number (selling price of a car)

![[Pasted image 20230519130245.png]]

# 1. Getting the data ready

Three main things we have to do:

    1. Split the data into features and labels (usually 'X' & 'y')
    2. Filling (also called imputing) or disregarding missing values
    3. Converting non-numerical values to numerical values (also called feature encoding)

# 2. Choosing the right Estimator/ algorithm for your problem

some things to note:

- Sklearn refer to machine learning models, algorithm as estimators
- `classification problem` - predicting a category (heart disease or not)
  - Something you'will see `clf` (classifier) used as a classification estimator
- `Regression problem` - predicting a number (selling price of a car)

- Step 1 - Check the sklearn machine learning map... https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# 3. Fit the model / algorithm on our data and use it to make predictions

### 3.1 Fitting the model to the data

Different names for:

- `X` = features, features variables, data
- `y` = labels, targets, target variables](<3. Fit the model to data and using it to make predictions
  Now you've chosen a model, the next step is to have it learn from the data so it can be used for predictions in the future.

If you've followed through, you've seen a few examples of this already.

3.1 Fitting a model to data
In Scikit-Learn, the process of having a machine learning model learn patterns from a dataset involves calling the fit() method and passing it data, such as, fit(X, y).

Where X is a feature array and y is a target array.

Other names for X include:

Data
Feature variables
Features
Other names for y include:

Labels
Target variable
For supervised learning there is usually an X and y. For unsupervised learning, there's no y (no labels).

Let's revisit the example of using patient data (X) to predict whether or not they have heart disease (y).>)

```py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# setup random seed
np.random.seed(42)


# make the data

x=heart_disease.drop("target",axis=1) # features
y=heart_disease["target"] # labels

# split into train/test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# choose the right model and hyperparameters

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)

# we will keep the default hyperparameters

# fit the model to the data

model.fit(x_train,y_train)

# evaluate the model on the test data
model.score(x_test,y_test)


```

![[Pasted image 20230521121320.png]]
What's happening here?

Calling the `fit()` method will cause the machine learning algorithm to attempt to find patterns between `X` and `y`. Or if there's no `y`, it'll only find the patterns within `X`.

Let's see `X`.

Passing `X` and `y` to `fit()` will cause the model to go through all of the examples in `X` (data) and see what their corresponding `y` (label) is.

How the model does this is different depending on the model you use.

Explaining the details of each would take an entire textbook.

For now, you could imagine it similar to how you would figure out patterns if you had enough time.

You'd look at the feature variables, `X`, the `age`, `sex`, `chol` (cholesterol) and see what different values led to the labels, `y`, `1` for heart disease, `0` for not heart disease.

This concept, regardless of the problem, is similar throughout all of machine learning.

**During training (finding patterns in data):**

A machine learning algorithm looks at a dataset, finds patterns, tries to use those patterns to predict something and corrects itself as best it can with the available data and labels. It stores these patterns for later use.

**During testing or in production (using learned patterns):**

A machine learning algorithm uses the patterns its previously learned in a dataset to make a prediction on some unseen data.

### 3.2 Making predictions using a machine learning model

Now we've got a trained model, one which has hoepfully learned patterns in the data, you'll want to use it to make predictions.

Scikit-Learn enables this in several ways. Two of the most common and useful are [`predict()`](https://github.com/scikit-learn/scikit-learn/blob/5f3c3f037/sklearn/multiclass.py#L299) and [`predict_proba()`](https://github.com/scikit-learn/scikit-learn/blob/5f3c3f037/sklearn/linear_model/_logistic.py#L1617).

Let's see them in action.

# Use a trained model to make predictions

```py
clf.predict(X_test)
```

```
array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0])
```

Given data in the form of `X`, the `predict()` function returns labels in the form of `y`.

It's standard practice to save these predictions to a variable named something like `y_preds` for later comparison to `y_test` or `y_true` (usually same as `y_test` just another name).

# 4. Evaluating a machine learing model[](http://localhost:8888/notebooks/scikit-learn.ipynb#4.-Evaluating-a-machine-learing-model)

Three ways to Evalute scikit-Learn models/estimators:

1.Estimator's build-in `score()` method

2.The `scoring` parameter

3.Problem-specific metric functions

you can read more about it here: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

### 4.1 Evaluate model using `scoring` parameter[](http://localhost:8888/notebooks/scikit-learn.ipynb#4.1-Evaluate-model-using-scoring-parameter)

```py
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

​

x=heart_disease.drop("target",axis=1)

y=heart_disease["target"]

clf=RandomForestClassifier()

​

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

​

# fit the model

​

clf.fit(x_train,y_train)

​

# score

clf.score(x_test,y_test)

​

Out[17]:

0.8524590163934426
```

## Cross-Validation

![[sklearn-cross-validation 1.png]]

### we split the data to test

![[Pasted image 20230521195131.png]]

![[Pasted image 20230521195102.png]]
this is really best practice because we test data from different split

## 4.2.1 Classification model evaluation metrics

1. Accuracy
2. Area under ROC curve
3. Confusion matrix
4. Classification report

### Area under the ROC curce

**Area under the reciver operation characteristic curve (AUC/ROC)**'

ROC curves are a comparison of a model's true positive rate (tpr) versus a models false positive rate (fpr).

`tpr`=true positive rate = model predicts 1 when truth is 1

`fpr`=false positive rate = model predicts 1 when truth is 0

- `True positive` = model predicts 1 when truth is 1
- `False positive` = model predicts 1 when truth is 0
- `True negative` = model predicts 0 when truth is 0
- `False negative` = model predicts 0 when truth is 1

### Reading Extension: ROC Curve + AUC

When you first encounter them, ROC Curve and AUC (area under curve) metrics can be a little confusing. But not to worry, with a little practice, they'll start to make sense.

In a nutshell, what you should remember is:

- `ROC` curves and `AUC` metrics are evaluation metrics for binary classification models (a model which predicts one thing or another, such as heart disease or not).
- The `ROC` curve compares the true positive rate (tpr) versus the false positive rate (fpr) at different classification thresholds.
- The `AUC` metric tells you how well your model is at choosing between classes (for example, how well it is at deciding whether someone has heart disease or not). A perfect model will get an AUC score of 1.

For more information on these metrics, bookmark the following resources and refer to them when you need:

- [ROC and AUC, Clearly Explained!](https://www.youtube.com/watch?v=4jRBRDbJemM) by StatQuest
- [ROC documentation in Scikit-Learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) (contains code examples)
- [How the ROC curve and AUC are calculated](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) by Google's Machine Learning team

**Confusion matrix**

A Confusion matrix is a quick way to compare the labels predicts by the model with the actual labels it was supposed to predict.

In essence, giving you an idea of where the model is getting confused.

![[sklearn-confusion-matrix-anatomy.png]]

### Classification model evaluation metrics

1. R^2 (pronounced r-squared) or coefficient of determination

2. Mean absolute error (MAE)

3. Mean squared error (MSE)

```py
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

np.random.seed(42)

x=heart_disease.drop("target",axis=1)
y=heart_disease["target"]
clf=RandomForestClassifier()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# fit the model

clf.fit(x_train,y_train)

# score
clf.score(x_test,y_test)

# make predictions
y_preds=clf.predict(x_test)

print("Classifier metric on the test set")
print(f"Accuracy:{accuracy_score(y_test,y_preds)*100:.2f}%")
print(f"Precision:{precision_score(y_test,y_preds)*100:.2f}%")
print(f"Recall:{recall_score(y_test,y_preds)*100:.2f}%")

print(f"F1:{f1_score(y_test,y_preds)*100:.2f}%")

#Result

Classifier metric on the test set Accuracy:85.25%
Precision:84.85%
Recall:87.50%
F1:86.15%
```

## Regression model evaluation metrics

```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

np.random.seed(42)

# get the data ready
x=house_df.drop("target",axis=1)
y=house_df["target"]

# split into train/test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# fit the model

model=RandomForestRegressor()

model.fit(x_train,y_train)

# predict

y_preds=model.predict(x_test)

# evaluate the model

print(f"mean absolute error:{mean_absolute_error(y_test,y_preds)*100:.2f}")
print(f"mean squared error:{mean_squared_error(y_test,y_preds)*100:.2f}")
print(f"r2 score:{r2_score(y_test,y_preds)*100:.2f}")

#Result
mean absolute error:32.66
mean squared error:25.34
r2 score:80.66
```

MAE- the average of the absolute differences between predictions and actual values. It gives you an idea of how wrong your model predictions are.

MSE- the squred average of the difference between the predictions and actual values. Squaring the errors removes negative errors. It gives you a rough idea of how far your predictions are from the actual values.

## 5. Imporving a model

first predictions = baseline predictions
first model = baseline model

From a data Perspective:

- Could we collect more data? (generally, the more data, the better)
- Could we improve our data?

From a model perspective:

- Is there a better model we could use?
- Could we improve the current model?

Hyperparameters vs. Parameters

`Parameters` = model find these patterns in data

`Hyperparameters` = settings on a model you can adjust to (potentially) improve its ability to find patterns

Three ways to adjust `hyperparameters`:

1. By hand

2. Randomly with RandomSearchCV

3. Exhaustively with GridSearchCV

![[sklearn-train-valid-test-annotated.png]]

![[Pasted image 20230523111733.png]]

### 5.1 Tuning hyperparameters by hand

We're going to try and adjust:
* `max_depth`
* `max_features`
* `min_samples_leaf`
* `min_samples_split`
* `n_estimators`
```py
def evaluate_preds(y_true,y_preds):
    """
    Performs evaluation comparison on y_true labels vs y_pred labels
    """

    accuracy=accuracy_score(y_true,y_preds)
    precision=precision_score(y_true,y_preds)
    recall=recall_score(y_true,y_preds)
    f1=f1_score(y_true,y_preds)
    metric_dict={"accuracy":round(accuracy,2),
                 "precision":round(precision,2),
                 "recall":round(recall,2),
                 "f1":round(f1,2)}

    print(f"Acc:{accuracy*100:.2f} %")
    print(f"Precision:{precision* 100:.2f} %")
    print(f"Recall:{recall* 100:.2f} %")
    print(f"F1:{f1*100:.2f} %")

    return metric_dict
```


```py
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# shuffle the data

heart_disease_shuffled=heart_disease.sample(frac=1)

# split into x/y

x=heart_disease_shuffled.drop("target",axis=1)

y=heart_disease_shuffled["target"]


# split into train/test/validation

train_split=round(0.7*len(heart_disease_shuffled)) # 70% of data for training
valid_split=round(train_split+0.15*len(heart_disease_shuffled)) # 15% of data for validation
# 70% + 15% = 85% of data for training and validation
x_train,y_train=x[:train_split],y[:train_split] # from 0 to train_split
 
x_valid,y_valid=x[train_split:valid_split],y[train_split:valid_split] # from train_split to valid_split

x_test,y_test=x[valid_split:],y[valid_split:] # from valid_split to end

len(x_train),len(x_valid),len(x_test)

clf=RandomForestClassifier()

clf.fit(x_train,y_train)

# make baseline predictions
y_preds=clf.predict(x_valid)

# evaluate the classifier on validation set

baseline_metrics=evaluate_preds(y_valid,y_preds)

baseline_metrics

```

### 5.2 Hyperparameter tuning with RandomizedSearchCV

```py
from sklearn.model_selection import RandomizedSearchCV,train_test_split

grid={"n_estimators":[10,100,200,500,1000,1200],
        "max_depth":[None,5,10,20,30],
        "max_features":["auto","sqrt"],
        "min_samples_split":[2,4,6],
        "min_samples_leaf":[1,2,4]}
np.random.seed(42)

np.random.seed(42)

# shuffle the data

heart_disease_shuffled = heart_disease.sample(frac=1)

# split into x/y

x = heart_disease_shuffled.drop("target", axis=1)

y = heart_disease_shuffled["target"]

# split into train/test/

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
clf = RandomForestClassifier(n_jobs=1)

# setup RandomizedSearchCV

rs_clf = RandomizedSearchCV(estimator=clf,
                                param_distributions=grid, # what we created above
                                n_iter=5, # number of models to try
                                cv=5, # cross validation
                                verbose=2 # print out results
                                )

# fit the RandomizedSearchCV version of clf

rs_clf.fit(x_train, y_train) # it will make validation set automatically

# find the best parameters

rs_clf.best_params_

```

### 5.3 Hyperparameter tuning with GridSearchCV

```py
from sklearn.model_selection import GridSearchCV

grid_2 = {"n_estimators": [100, 200, 500],
            "max_depth": [None],
            "max_features": ["auto", "sqrt"],
            "min_samples_split": [6],
            "min_samples_leaf": [1, 2]}
np.random.seed(42)

# shuffle the data

heart_disease_shuffled = heart_disease.sample(frac=1)

# split into x/y 

x = heart_disease_shuffled.drop("target", axis=1)

y = heart_disease_shuffled["target"]

# split into train/test/

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

clf = RandomForestClassifier(n_jobs=1)

# setup GridSearchCV

gs_clf = GridSearchCV(estimator=clf,
                        param_grid=grid_2, # what we created above
                        cv=5, # cross validation
                        verbose=2 # print out results
                        )

# fit the GridSearchCV version of clf

gs_clf.fit(x_train, y_train) # it will make validation set automatically

# find the best parameters

gs_clf.best_params_

# evaluate the randomized search RandomForestClassifier model

gs_y_preds = gs_clf.predict(x_test)

# evaluate the predictions

gs_metrics = evaluate_preds(y_test, gs_y_preds)

gs_metrics

# compare the different models metrics

compare_metrics = pd.DataFrame({"baseline": baseline_metrics,
                                
                                "random search": rs_metrics,

                                "grid search": gs_metrics})

compare_metrics.plot.bar(figsize=(10, 8));
```


# compare the different models metrics
![[Pasted image 20230529114516.png]]


## 6 . Saving and loading trained machine learning models

Two ways to save and load machine learning models:

1. With Python's `pickle` module
2. With the `joblib` module


```py
# 6. Saving and loading trained machine learning models

import pickle

# save an existing model to file

pickle.dump(gs_clf, open("gs_random_forest_model_1.pkl", "wb"))

# load a saved model

loaded_pickle_model = pickle.load(open("gs_random_forest_model_1.pkl", "rb"))

# make some predictions
["hear disease " if i ==1 else "not Heart Disease" for i in loaded_pickle_model.predict(x_test)]


# make some predictions

pickle_y_preds = loaded_pickle_model.predict(x_test)

evaluate_preds(y_test, pickle_y_preds)

```