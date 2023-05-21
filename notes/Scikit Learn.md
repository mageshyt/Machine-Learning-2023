`Refresher` : what is Machine Learning
	normally we developer write `function` and give `inputs`  to the machine that perform the `task` and give `output`

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
    -   [Scikit-Learn documentation/user guide](https://scikit-learn.org/stable/user_guide.html) - the most extensive resource you'll find for Scikit-Learn information.
    -   [Stack Overflow](https://stackoverflow.com/) - this is the developers Q&A hub, it's full of questions and answers of different problems across a wide range of software development topics and chances are, there's one related to your problem.

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

* Sklearn refer to machine learning models, algorithm as estimators
* classification problem - predicting a category (heart disease or not)
	* Something you'will see `clf` (classifier) used as a classification estimator
 * Regression problem - predicting a number (selling price of a car)

![[Pasted image 20230519130245.png]]


# 1. Getting the data ready

Three main things we have to do:

    1. Split the data into features and labels (usually 'X' & 'y')
    2. Filling (also called imputing) or disregarding missing values
    3. Converting non-numerical values to numerical values (also called feature encoding)



# 2. Choosing the right Estimator/ algorithm for your problem

some things to note:

* Sklearn refer to machine learning models, algorithm as estimators
* `classification problem` - predicting a category (heart disease or not)
	* Something you'will see `clf` (classifier) used as a classification estimator
 * `Regression problem` - predicting a number (selling price of a car)

* Step 1 - Check the sklearn machine learning map... https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html


# 3. Fit the model / algorithm on our data and use it to make predictions

### 3.1 Fitting the model to the data

Different names for:
* `X` = features, features variables, data
* `y` = labels, targets, target variables](<3. Fit the model to data and using it to make predictions
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