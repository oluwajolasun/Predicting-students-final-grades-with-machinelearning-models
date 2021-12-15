  # Background on code files
  - Please consult `Oluwajolasun_Jaiyesimi_FinalProject.ipynb` for all sections and data for this project. Results, visually represented data and observations within the notebook
    - jupyter notebook using python and sklearn
  - Dataset from UCI dataset Visit https://archive.ics.uci.edu/ml/datasets/Student+Performance (33 features, 395 samples) when using only the mathematics data
  - Models used:
    - Scikit Learn classifiers: Logistic Regression, SVC, Random Forest Classifier, Gradient Boosting Classifier, DecisionTreeClassifier, BernoulliNB

  # How to run
  Please see links below for installing the necessary dependencies for this notebook:
  - Dependencies:
    - Sklearn:  https://scikit-learn.org/stable/install.html
    - Pandas:   https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
    - Numpy:    https://numpy.org/install/
    - Seaborn:  https://seaborn.pydata.org/installing.html
    - Matplotlib: https://pypi.org/project/matplotlib/
    - Mglearn: https://pypi.org/project/mglearn/

  - Ensure the `student-mat.csv` is in the working directory of the notebook when running the notebook. *Available in `data` folder of the repo*
  
  After those are installed, simply run all the cells in the order of the notebook and you should be good to go!
  
  However, some of the grid searches might take a while to run.

# Results

Below are the final scores for the top three models for the dataset without `G1` and `G2` and with `G1` and `G2` after tuning them to their best hyperparameters and testing on the unseen test data:
However, due to the class imbalance, I will be using the macro average results from the classification report.

## Final Results For Data Without `G1` and `G2`
### LogisticRegression
* Precision = 0.68
* Recall = 0.63
* F1-score = 0.63
* FP = 7
* FN = 21

### SVC
* Precision = 0.62
* Recall = 0.61
* F1-score = 0.61
* FP = 11
* FN = 25

### GradientBootingClassifier
* Precision = 0.55
* Recall = 0.54
* F1-score = 0.53
* FP = 14
* FN = 19

## Final Results For Data With `G1` and `G2`
### LogisticRegression
* Precision = 0.93
* Recall = 0.93
* F1-score = 0.93
* FP = 2
* FN = 3

### RandomForestClassifier
* Precision = 0.93
* Recall = 0.94
* F1-score = 0.93
* FP = 4
* FN = 1

### GradientBoostingClassifier
* Precision = 0.94
* Recall = 0.92
* F1-score = 0.93
* FP = 1
* FN = 4

# Interpretation
## Best model
It is clear that the results gotten when `G1` and `G2` were not included yielded low accuracy performances by the models, Their f1-scores are seperated by a 10% margin with the LogisticRegression classifier having the best f1-score of 0.63 and SVC being very close with a f1-score of 0.61, I will also look at the false negative and false positive predictions for these two models to make further conclusion. For these scenario, both false positive and negative predictions for the LogisticRegression model is lower than SVC further concluding that the `LogisticRegression` model works best when `G1` and `G2` are not included as features in our dataset.

With the feature `G1` and `G2` included, we get very interesting results of the models having the exact f1-score of 0.93 which is quite impresive. However, I still cannot tag the best model with the entire dataset when they all have the exact f1-score. I will look the the false negative and positive predictions as I did in the previous test, to select the best model. The RandomForestClassifier has more false positive than false negative which means more students falsey passed, which means student who do not deserve the grade get a pass score which would not be a good scenario based of the situaition (Ranking a school based on student grades, Students getting promoted to the next class with limited seat space, etc.). Similarly, with the GradientBoostingClassifier has more false negative than false positive which means more students falsey failed when they did actually pass, which means student who actually deserve the pass grade were failed which would not be a good scenario based of the situaition (Students relying on their math final score grade to bump up their overall GPA, etc.). With LogisticRegression we get a balance of the predicted False negative and positive of 3 and 2 respectively, which is better than having more falsey on one side of the class than the other. I will conlude that `LogisticRegression` model also works best when `G1` and `G2` are also included as features in our dataset.

# Reflection
I learned quite a lot from this mini project. Having the mixture of categorical and numerical data helped me get more of a feel for what is required for encoding features in datasets (preprocessing). I also feel much more comfortable grid searching and interpreting the results from those grid searches. 

With regards to the proposal, I stated I will use lab 2, but I ended using both lab 2 and lab 3 for this mini project, I followed the general structure/flow of that lab to explore this data and train machine learning models for the final grades prediction. I also stated using both classification and regression models for the mini project which would be more interesting comparing if classification or regression works better for this dataset, I had already created a scoring function to get the negative root mean square error which is a score for regressor models and also decided to use the default `G3` column in the initial dataset as my target vector which I called `y_reg`. However adding regressor models will deviate from this project being more than a mini project. So definitely using regressor models for the dataset will be something to definitely do in the near future.

In summary, I feel I explored and created 3 models that have good performance at predicting students final math grade, provided we can collect all the data used as features for the dataset, including `G1` and `G2`. This was a great learning experience and I look forward to exploring more datasets and machinelearning models in the future!
