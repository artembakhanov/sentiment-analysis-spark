# Assignment №2. Introduction to Big Data. Stream Processing with Spark

**Team Members:**

* Artem Bakhanov
* Dmitry Podpryatov
* Kamil Kamaliev
* Marina Nikolaeva

## The Problem

**What:**
Given a stream of tweets, analyze their sentiment using Apache Spark. The task is to build a model that will predict
whether a given tweet possesses a positive or negative emotions.

**Why:** First of all, to practice our Spark skills. Second of all, this can be useful for companies because they can
get customers' opinion about their products &mdash; and thus fix what's bad or understand what people like. 

## The Solution

We divided the work into stages and created a pipeline that can be summarized as follows:

![Pipeline](https://i.imgur.com/aTvz0KO.jpg)

**1. Selecting a dataset**
 
We looked for labeled data on the internet, selected datasets that we thought would suit us the most, and merged
them together.
Check out [references](#references) for the sources that we selected.

## What can be Improved

## Testing

## Output

## References

* [Assignment description](https://hackmd.io/@BigDataInnopolis/ryWL1HiKS#Writing-the-Report)

* Labeled datasets:
    * [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) &mdash; binary classified sentiment
    dataset that contains a total of 50000 movie reviews;
    * [UCI Sentiment Labelled Senteces](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) &mdash;
    binary classified reviews from [imbd.com](https://imbd.com), [amazon.com](https://amazon.com), and [yelp.com](https://yelp.com);
    * [Twitter Sentiment](https://www.kaggle.com/c/twitter-sentiment-analysis2/data) &mdash; sentimentally labeled twitter messages.

* [Reasons for twitter sentiment analysis](https://monkeylearn.com/blog/sentiment-analysis-of-twitter/#:~:text=Sentiment%20analysis%20is%20the%20automated,are%20talking%20about%20their%20brand.)
