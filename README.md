# Assignment №2. Introduction to Big Data. Stream Processing with Spark

**DS-02 Team Members and Roles:**

* **Artem Bakhanov**

Team management, stream reading, model creation and tuning (linear SVC), code refactoring

* **Dmitry Podpryatov**

Model creation (random forest) and tuning (all but SVC), report

* **Kamil Kamaliev**

Model creation (logistic regression), stream, data preprocessing and merging datasets, report

* **Marina Nikolaeva**

Model evaluation (scripts and labeling), support internationality of the team, report

## The Problem

**What:**
Given a stream of tweets, analyze their sentiment using Apache Spark. The task is to build a model that will predict
whether a given tweet possesses a positive or negative emotions.

**Why:** First of all, to practice our Spark skills. Second of all, this can be useful for companies because they can
get customers' opinion about their products &mdash; and thus fix what's bad or understand what people like. 

## The Solution

We divided the work into stages and created a pipeline that can be summarized as follows:

![Pipeline](images/pipeline.jpg)

**1. Selecting data**
 
We looked for labeled data on the internet, selected datasets that we thought would suit us the most, and merged
them together.
Check out [references](#references) for the sources that we selected. Basically, we were looking for a pair
(sentiment, sentence / paragraph / text) for our classification models.

You can find `.ipynb` notebook where the datasets were merged together in the `merging datasets` folder.
The result of the merge is situated in the `data` folder. The data was split on the train and test at this point,
so we will not have to do it at the preprocessing stage.

**2. Preprocessing**

Before feeding the data to the classification models, we need to preprocess it. Since both our data and tweets from the
stream have similar format, the preprocessing module will be the same for the dataset and for the stream.

The preprocessing consists of several part:

    1. Get rid of unimportant information such as tags (@elonmusk, #twitter), hyperlinks (https://google.com), 
    html tags (<br /><br />), and repeated symbols (amaaaazing).
    
    2. Tokenization - split text into words
    
    3. Remove "stopwords" - words that do not influence the sentiment of the text (they, to, be, because, that, etc.)

    4. Transform words to vector model so that it can be fed into the classification model (Word2Vec)
    
As it was mentioned before, we have already split the data into train and test samples, so we do not have to do it here.
The preprocessing code is situated at the `src/main/scala` folder within `Preprocessing.scala` and `TrainWord2Vec.scala`.

**3. Classification**

After the data is preprocessed, it is fed into the classification models. Among all classification models we have chosen
[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression),
[Ranodm Forest](https://en.wikipedia.org/wiki/Random_forest), and [Support Vector Clustering (SVC)](https://en.wikipedia.org/wiki/Support_vector_machine).
We also tried to implement [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier, however it accepted only vectors with nonnegative values, so we got to
apply scaling which would mess up the `Word2Vec`, so we rejected this idea.

`ml` and `mllib` packages contain all listed classifiers. We will tune the models and compare their performance on the data.
Hyperparameters were tuned with cross validation and grid search for each classifier. Below are the grids and the best
parameters for each model. For detailed description of the hyperparameters check out links in the [references](#references) at the end.

**Logistic Regression**

| Parameter | Grid | Best Value |
| :--- | :---: | :---: |
| `elasticNetParam` | 0, 0.5, 0.8, 1 | 0.5 |
| `fitIntercept` | true, false | true |
| `maxIter` | 1000 | 1000 |
| `regParam` | 0, 0.1, 0.2 | 0.1 |
| `threshold` | 0.5 | 0.5 |

**Random Forest**

| Parameter | Grid | Best Value |
| :--- | :---: | :---: |
| `impurity` | entropy, gini | gini |
| `maxDepth` | 3, 5, 7 | 7 |
| `numTrees` | 20, 40 | 40 |

**SVC**

| Parameter | Grid | Best Value |
| :--- | :---: | :---: |
| `aggregationDepth` | 2, 3 | 2 |
| `maxIter` | 100, 150 | 150 |
| `threshold` | 0.4, 0.5, 0.6 | 0.6 |
| `regParam` | 0, 0.1, 0.2 | 0 |

We can assess the quality of our models using [F1 Score](https://en.wikipedia.org/wiki/F1_score). Below is the table with
models' F1 Score:

| Classifier | F1 Score |
| :--- | :---: |
| Logistic Regression | 0.721 |
| Random Forest | 0.718 |
| SVC | 0.586 |

The code for classifiers is situated in the `Classifiers.scala` (cross validation and grid search) and in the
`TrainClassifier.scala` the best parameters have been set explicitly to save time.

**4. Streaming**

So, we have the classifiers trained and ready to predict sentiments. Now we collect the tweets from the stream and
preprocess them as we did with the dataset. TODO

**5. Classify Tweets**

Now, we have our classifiers trained and tweets preprocessed. The only thing remaining is to predict the sentiment of
these tweets and assess the quality of our models absolutely and relative to each other using metrics.

## What can be Improved

## Testing

## Output

## Conclusion    

## References

* [Assignment description](https://hackmd.io/@BigDataInnopolis/ryWL1HiKS#Writing-the-Report)

* Labeled datasets:
    * [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) &mdash; binary classified sentiment
    dataset that contains a total of 50000 movie reviews;
    * [UCI Sentiment Labelled Senteces](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) &mdash;
    binary classified reviews from [imbd.com](https://imbd.com), [amazon.com](https://amazon.com), and [yelp.com](https://yelp.com);
    * [Twitter Sentiment](https://www.kaggle.com/c/twitter-sentiment-analysis2/data) &mdash; sentimentally labeled twitter messages.

* [Reasons for twitter sentiment analysis](https://monkeylearn.com/blog/sentiment-analysis-of-twitter/#:~:text=Sentiment%20analysis%20is%20the%20automated,are%20talking%20about%20their%20brand.)

* Classifiers:
    * [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression)
    and [Parameters](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/classification/LogisticRegression.html)
    * [Ranodm Forest](https://en.wikipedia.org/wiki/Random_forest) and [Parameters](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/classification/RandomForestClassifier.html)
    * [Support Vector Clustering (SVC)](https://en.wikipedia.org/wiki/Support_vector_machine) and [Parameters](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/classification/LinearSVC.html)
    
* Classification Metrics:

    * [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
    * [F1 Score](https://en.wikipedia.org/wiki/F1_score)
