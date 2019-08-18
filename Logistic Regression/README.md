# Survival Prediction on Titanic

This is an implementation of Logistic Regression for prediction of survival on the popular Titanic dataset on Kaggle

training Data pre-processing:

* The column Cabin was dropped since it has a lot of missing values although information could be extracted from it, but that’s more hard work than gained.

* The missing values of age feature were filled using the mean of all ages within the dataset.

* I have used min-max scaling for normalisation on the data to bring it in the same range.
