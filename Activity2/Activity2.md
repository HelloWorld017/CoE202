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
|       0.8036        |      0.9102     |          0.9697          |

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
The comparison accuracy have been taken from 8-chunked cross validation.
So the "Before" accuracy is lower than the value of problem 1.

Used classifier is SVM Classifier: `SVC(probability=True, gamma='auto')`

#### #1 Reviving FamilySize
Although there is some differences between family of four and family of ten but they're considered same as it is binary.
So I have revived family size and applied normalization to FamilySize.

##### Comparison
|        |      Accuracy     |
|--------|-------------------|
| Before | 0.7285 (± 0.0706) |
| After  | 0.7274 (± 0.0686) |

As the mean accuracy have been decreased, it won't be used.

#### #2 Embarked to one-hot
Originally embarked was a numeric value: 0, 1, 2  
But I have mapped this into one-hot vector: [1, 0, 0], [0, 1, 0], [0, 0, 1]  
as numerical relation doesn't hold between Q, C, S.

($2C \neq S$)

##### Comparison
|        |      Accuracy     |
|--------|-------------------|
| Before | 0.7285 (± 0.0706) |
| After  | 0.7353 (± 0.0778) |

#### #3 Parsing Cabin
Although many cabin are empty, as cabin means position in ship, it might worth reviving.

First, we split cabins by whitespace.
There are A-G cabins except the one `T` cabin.  
So, we'll remove the `T` cabin.

Second, we'll make it to vector with length 7(A~G). If a customer has some cabins in a deck, it will be 1. Otherwise it will be 0.

##### Comparison
|        |      Accuracy     |
|--------|:-----------------:|
| Before | 0.7285 (± 0.0706) |
| After  | 0.7421 (± 0.0858) |

#### #4 Title to one-hot
Originally title was a numeric value: 0, 1, ..., 5  
But I have mapped this into one-hot vector as numerical relation doesn't hold between titles.

##### Comparison
|        |      Accuracy     |
|--------|:-----------------:|
| Before | 0.7285 (± 0.0706) |
| After  | 0.7297 (± 0.0854) |

#### #5: Normalize Age, Fare
As Age and Fare values are relatively larger than other values, we normalize it to $[0, 1]$

##### Comparison
|        |      Accuracy     |
|--------|:-----------------:|
| Before | 0.7285 (± 0.0706) |
| After  | 0.8204 (± 0.0611) |

<!--
#### #6: Reviving tickets
This regexp can match tickets except `LINE`: `([A-Z0-9.\/ ]+? )?\d+`.  
For `LINE`, we drop the ticket info.  

Then this can be splitted into two parts: prefix and number

As prefix represents `Embarked`, we drop it. (STON/O2 ->S, A/5. -> S, PC -> C and more...)  

The number can be used to fill cabin. There are some people with same ticket number.
They're a group so if one of them have a cabin number, we can assume others also have same cabin number.  

* Although there are some exception (`PC 17485` have `E36` and `A20`)
but most of them have same cabin number or same deck.

##### Comparison
-->

### Modifying classifier
