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

#### #1 Reviving Parch, SibSp
I mapped Parch, SibSp to Protector and Protectee by its age and sex.

The rule is stated below:
* If Age is less than 16 or greater than 70, the Parch becomes Protector and SibSp becomes Protectee.  
* Else if Age is greater than 30, the Parch becoms Protectee.  
If it is male and has SibSp the Protectee is increased by 1 (Spouse).  
* Else (16 <= Age <= 30) the Parch becomes Protector but SibSp is just dropped.

##### Comparison
|        |      Accuracy     |
|--------|-------------------|
| Before | 0.7285 (± 0.0706) |
| After  | 0.7386 (± 0.0673) |

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
As Age, Fare values are not in $[0, 1]$ and relatively larger than other values, we normalize it to $[0, 1]$

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
The comparison accuracy have been taken from 8-chunked cross validation.

#### SVC
##### Kernels
```
======= Kernel =======
Rbf Accuracy: 0.8194 (+/- 0.0454)
Polynomial2 Accuracy: 0.8047 (+/- 0.0336)
Polynomial3 Accuracy: 0.7566 (+/- 0.0536)
Polynomial4 Accuracy: 0.6679 (+/- 0.0625)
Sigmoid Accuracy: 0.7934 (+/- 0.0489)
```

##### C value
```
======= C Value =======
0.1 Accuracy: 0.7889 (+/- 0.0468)
0.5 Accuracy: 0.8069 (+/- 0.0305)
1.0 Accuracy: 0.8194 (+/- 0.0454)
2.0 Accuracy: 0.8194 (+/- 0.0454)
5.0 Accuracy: 0.8171 (+/- 0.0499)
```

##### Gamma
```
======= Gamma =======
0.025 Accuracy: 0.8182 (+/- 0.0484)
0.05 Accuracy: 0.8194 (+/- 0.0454)
0.1 Accuracy: 0.8172 (+/- 0.0641)
0.2 Accuracy: 0.8249 (+/- 0.0689)
```

So I uploaded test prediction of SVM with gamma 0.2, C value 2.0, rbf kernel  
I didn't used Cabin Modification as the mean accuracy decreased when used with another modifications.

It scored 0.78947, which ranked as #3172.
![Submission](./images/svm_submission.png)

#### RandomForest
##### N_estimators
```
======= N_estimators =======
16 Accuracy: 0.8104 (+/- 0.0613)
32 Accuracy: 0.7891 (+/- 0.0827)
64 Accuracy: 0.8014 (+/- 0.0569)
128 Accuracy: 0.7958 (+/- 0.0588)
```

But it can be changed easily because of random. I used 64 as it shows genenerally good score when every time I run.

##### max_depth
```
======= max_depth =======
2 Accuracy: 0.7990 (+/- 0.0361)
4 Accuracy: 0.8170 (+/- 0.0427)
8 Accuracy: 0.8316 (+/- 0.0595)
16 Accuracy: 0.8148 (+/- 0.0494)
```

##### min_samples_split
```
======= min_samples_split =======
Default Accuracy: 0.8350 (+/- 0.0659)
0.01 Accuracy: 0.8306 (+/- 0.0752)
0.05 Accuracy: 0.8283 (+/- 0.0663)
0.1 Accuracy: 0.8160 (+/- 0.0384)
0.5 Accuracy: 0.7867 (+/- 0.0432)
```

So I uploaded test prediction of RandomForest with max_depth 8, n_estimators 64, min_samples_split 0.01  
And I used all modification of preprocessing and it scored 0,78947, too, which ranked as #3172.

![Submission](./images/randomforest_submission.png)  

![Leaderboard](./images/submission_rank.png)
