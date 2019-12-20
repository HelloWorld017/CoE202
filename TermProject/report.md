# Term Project
CoE202(A) / 20190146 Kim Yohan

## Problem 1
### Preprocessing
I have slightly modified preprocessing code in order to preprocess not only products, but also brands, models and makers.

* For `model`:  
As some of `model`s are composed with many words, I have reused the code which are used to preprocess `product`s.

* For `brand`, `maker`:
I could easily notice that some of `brand` and `maker` informations are not given and written as `"상세참조"`.  
So I made it to skip preprocessing when it contains `상세` or `참조` in them.  
I have removed special characters like other fields and hashed it.

```py
maker = df['maker'][i]
maker_i = 0
if isinstance(maker, str) and not pd.isnull(maker) and maker and \
	"상세" not in maker and "참조" not in maker:

	# As maker is nearly one word, we can just use it
	maker = re_sc.sub('', maker)
	maker_i = hash(maker) % 100000 + 1
```

And saved x value as array of `[brands, models, model_counts, makers, products, product_counts, prices]`.

### Training
I have embedded `products`, `brands`, `models`, `makers` by reusing an Embedding layer.  
For `products` and `models`, like the base code, I also multiplied the counts to corresponding embedding vectors and added them.

For `brands`, `makers`, I just used the result of embedding vector.
Finally, I have concatenated them all and passed them to two `dense + bn + relu` layers.
Between the layers, I have flattened the values.

### Results
|          |  Original  | Modified |
|:--------:|:----------:|:--------:|
| Val Top1 |   0.7439   |  0.7541  |
| Val Top5 |   0.9065   |  0.9215  |

The two validation accuracy slightly increased.

## Problem 2
### Preprocessing
* For `images`, `prices`: Just used as-is

### Training
As `prices` needs normalization, I have added batch normalization layer to the prices.  
For `images`, it passes through the hidden layer, which is consisted of fully-connected, batch normalization, ReLU.  

And they are combined and passes through the hidden layer2 and output layer.

### Results
|          |  Original  | Modified |
|:--------:|:----------:|:--------:|
| Val Top1 |   0.7439   |  0.6886  |
| Val Top5 |   0.9065   |  0.9003  |

Unfortunately, the accuracy have been decreased slightly.

## Problem 3
For problem 3, I have rewrote the code.

### Idea
For things that marked as `NOT`, it is not implemented.
For things that marked as `DEC`, it is removed due to accuracy decrease.

* Combining data used in Problems 1, 2
* Improving text processing
  * It might contain hash collision. Find another way of mapping word to integer.
  * Removing word that is only one-time used. `NOT`
  * Instead of embedding sentence into a vector, it will use Bidirectional GRU & Time Distributed Dense `DEC`
* Ensemble Learning `NOT`
* Increasing epoch `NOT`
* Tuning Hyperparameters
