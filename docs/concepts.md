# Concepts

> Key concepts introduced in this module.

<!--
Only the first sentence/paragraph of h3 entries
are used for the integrated quiz.

Wrap code terms in double asterisks
rather than single backtics so they can be read aloud.
-->

## Machine Learning

Machine learning uses data to estimate patterns that can be applied
to observations beyond those used to build the model.

### Machine Learning Model

A **machine learning model** is a learned representation of patterns
in data that can be used to make predictions.

The model is built from data rather than from a fixed set
of manually written decision rules.

### Supervised Learning

**Supervised learning** builds a model using observations
for which the outcome to be predicted is already known.

The known outcomes provide examples the model can learn from.

Regression and classification are common supervised-learning tasks.

### Regression

**Regression** is a supervised-learning task
that predicts a numeric outcome.

Examples include predicting sales, temperature, duration,
energy use, or another measured quantity.

### Generalization

**Generalization** is a model's ability to perform well
on observations it did not use while learning.

A model that describes its training data well
but performs poorly on new observations does not generalize well.

## Training and Testing

A predictive model must be evaluated using data
that was not used to fit the model.

### Training Data

**Training data** is the portion of the available data
used to fit a machine learning model.

The model learns its fitted parameters from these observations.

### Test Data

**Test data** is held aside during training
and used afterward to evaluate the fitted model.

Keeping test observations separate provides a more useful check
of how the model may perform on new data.

### Train-Test Split

A **train-test split** divides available observations
into separate training and testing sets.

The training set is used to fit the model.

The test set is reserved for evaluation.

### Random State

A **random state** fixes the random choices made by an operation
so the same split or result can be reproduced.

Using the same random state makes comparisons between experiments
more consistent.

### Data Leakage

**Data leakage** occurs when information that should be unavailable
during training influences the fitted model.

For example, using test data while fitting a model
weakens the value of the later test evaluation.

## Linear Regression Terms

Linear regression models a numeric outcome
as a straight-line relationship with one or more features.

### Linear Regression

**Linear regression** estimates a straight-line relationship
between numeric inputs and a numeric outcome.

For one feature, the fitted relationship can be written as:

```text
predicted value = intercept + slope × feature value
```

The fitted line summarizes the pattern the model learned
from the training observations.

### LinearRegression

**LinearRegression** is the scikit-learn class
used in this project to create a linear regression model.

For example:

```python
model = LinearRegression()
```

Creating the object defines the kind of model to use.

The model has not learned from the data yet.

### Fit

Calling **fit()** makes a model learn its fitted parameters
from the training data.

For example:

```python
model.fit(X_train, y_train)
```

For linear regression, fitting estimates the coefficients
and intercept that best describe the training observations.

### Coefficient

A linear regression **coefficient** describes how much
the predicted outcome changes for a one-unit change in a feature.

In a model with one feature, the coefficient is the slope
of the fitted regression line.

A positive coefficient means the fitted line rises
as the feature increases.

A negative coefficient means the fitted line falls.

### Intercept

The **intercept** is the model's predicted outcome
when all feature values equal zero.

It determines where the fitted regression line
crosses the outcome axis.

The intercept may or may not have a useful real-world interpretation,
depending on whether zero is meaningful for the features.

### Predict

Calling **predict()** asks a fitted model
to estimate outcomes for supplied feature values.

For example:

```python
y_pred = model.predict(X_test)
```

Predictions for the test data can be compared
with the known test outcomes.

### Predicted Value

A **predicted value** is the outcome estimated by a fitted model
for a particular observation.

For linear regression, predicted values lie
on the fitted regression line or surface.

## Prediction Error

Predictions rarely match every observed value exactly.

The differences between observed and predicted values
provide evidence about model performance.

### Residual

A **residual** is the observed outcome
minus the outcome predicted by the model.

```text
residual = observed value - predicted value
```

A positive residual means the observed value
was greater than the prediction.

A negative residual means the observed value
was less than the prediction.

### Residual Plot

A **residual plot** displays prediction residuals
to help reveal patterns in model errors.

For a useful linear model, residuals should generally appear
without a strong systematic pattern.

Curvature or other structure can suggest that a straight line
does not capture an important pattern in the data.

### Overfitting

**Overfitting** occurs when a model learns the training data
too specifically and performs worse on new data.

Evaluation on held-out test data helps reveal this problem.

## Model Evaluation Terms

No single evaluation measure tells the complete story.

Numerical metrics and diagnostic plots provide different evidence
about how well a regression model performs.

### Baseline Model

A **baseline model** provides a simple reference
for judging whether a fitted model adds predictive value.

A useful model should generally improve on an appropriate baseline.

### R-Squared

**R-squared**, written R², measures how much of the variation
in the observed outcome is explained by a regression model.

A value closer to 1 generally indicates that the model
accounts for more of the observed variation.

A value near 0 indicates little improvement
over predicting the same reference value for every observation.

R² can also be negative on test data
when the fitted model performs poorly.

### RMSE

**Root mean squared error (RMSE)** summarizes
the typical size of prediction errors in outcome units.

RMSE gives larger errors more influence
because errors are squared before they are averaged.

Lower RMSE indicates smaller prediction errors
when comparing models on the same outcome and data.

### Model Evaluation

**Model evaluation** uses held-out data, metrics,
and diagnostics to judge how well a fitted model performs.

For regression, this project examines measures such as R² and RMSE
along with predictions and residuals.

A model should be judged by the combined evidence,
not by a single metric alone.

### Model Assessment

A **model assessment** is the analyst's judgment
about whether the fitted model is useful for the intended purpose.

The assessment considers test performance,
diagnostic evidence, limitations, and the analytical context.

A mathematically valid fitted model is not automatically
a useful model.

---

[◄ Back to Home](index.md)
