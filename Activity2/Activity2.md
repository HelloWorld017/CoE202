# Activity 2
CoE202(A) / 20190146 Kim Yohan

## Before improvement
----
### Problem 1.
**Compare the prediction accuracies of the 3 models.**

* Logistic Regression
```py
classifier = LogisticRegression()
```
* SVM Classifier
```py
classifier = SVC(probability=True) # has rbf kernel
```
* Random Forest Classifier
```py
classifier = RandomForestClassifier()
```

And accuracies are taken like this:
```py
accuracy = classifier.score(X_train, Y_train) * 100
```
$$accuracy(y, \hat{y}) = \frac{1}{n_{samples}} \sum^{n_{samples} - 1}_{i = 0} 1 (\hat{y_i} = y_i)$$

<center>

| Logistic Regression | SVM Classifier  | Random Forest Classifier |
|:-------------------:|:---------------:|:------------------------:|
|        80.36        |       91.02     |           96.97          |

</center>

----
### Problem 2.
**Draw the ROCs of the 3 models.**
```py
FPR, TPR, thresholds = roc_curve(Y_train, Y_train_pred)

plt.plot(FPR, TPR)
```

<center>

| Logistic Regression | SVM Classifier  | Random Forest Classifier |
|:-------------------:|:---------------:|:------------------------:|
|![Logistic Regression](./images/roc_logisticregression.png)|![SVM Classifier](./images/roc_svmclassifier.png)|![Random Forest Classifier](./images/roc_randomforest.png)|

</center>

----

### Problem 3
**Obtain the AUCs(Areas Under Curves). which model performs the best?**
```py
AUC = roc_auc_score(Y_train, Y_train_pred)
```

<center>

| Logistic Regression | SVM Classifier  | Random Forest Classifier |
|:-------------------:|:---------------:|:------------------------:|
|         0.87        |       0.94      |            0.99          |

</center>

----

## Improvements
### Modifying preprocessing
#### Embarked to one-hot
##### Comparison

#### Title to one-hot

##### Comparison

#### Reviving family size
Although there is some differences between family of four and family of ten but they're considered same as it is binary.
So I have revived family size.

##### Comparison

#### Reviving cabin
Although many cabin are empty, as cabin means position in ship, it might worth reviving.

First, we split cabins by whitespace.
There are A-G cabins except the one `T` cabin.  
So, we'll remove the `T` cabin.

Second, we'll make it to vector with length 7(A~G). If a customer has some cabins in a deck, it will be 1. Otherwise it will be 0.

##### Comparison

#### Reviving tickets
There are some types of tickets.
1. `\d+` (ex: 364500)
2. `[A-Z0-9.]+ \d+` (ex: PC 17756)
3. `[A-Z0-9.]+\/[A-Z0-9.]+ \d+` (ex: STON/O2. 3101290)

##### Comparison

### Modifying classifier
