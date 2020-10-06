import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2VecModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.FloatType

class Preprocessing(spark: SparkSession, vec_size: Int, min_count: Int) {

  // reducing 3 or more repeating symbols to 2
  def reduc_repit_symb = (text: String) => {

    val reg = "(\\w)\\1+".r
    var res = text
    for(m <- reg.findAllIn(res)) {
      if (m.length > 2) {
        res = res.replaceAll(m, s"${m(0)}${m(0)}")
      }
    }
    res
  }

  // preprocessing for both train and test data
  def prep(dataFrame: DataFrame): DataFrame = {

    val no_repited_symb = spark.udf.register("no_repited_symb", reduc_repit_symb)
    var df = dataFrame
    // removing usernames and hastags
    df = df.withColumn("Text", regexp_replace(df.col("Text"), "(#|@)\\w+", " "))
    df = df.withColumn("Text", regexp_replace(df.col("Text"), "http[^\\s]+", " "))
    df = df.withColumn("Text", regexp_replace(df.col("Text"), "<.*>", " "))
    df = df.withColumn("Text", no_repited_symb(df.col("Text")))

    // defining tokenizer
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("Text")
      .setOutputCol("Tokens")
      .setPattern("\\w+")
      .setGaps(false)

    // tokenizing the text
    val tokenized = regexTokenizer.transform(df)

    // against not no
    val stopwords = StopWordsRemover.loadDefaultStopWords("english")
      .filter(!Array("against", "no", "not").contains(_))

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("Tokens")
      .setOutputCol("Filtered")
      .setStopWords(stopwords)

    df = stopWordsRemover.transform(tokenized)

    df
  }

  // preprocessing for testing data
  def prep_test(df_test: DataFrame): DataFrame = {

    var df = prep(df_test)

    val word2VecModel = Word2VecModel.load("word2VecModel")

    df = word2VecModel.transform(df)

    df = df.drop("Tokens", "Filtered")

    df = df.withColumnRenamed("Sentiment", "label")
    df = df.withColumnRenamed("Vec", "features")
    df = df.withColumn("labeltmp", df.col("label").cast(FloatType))
      .drop("label")
      .withColumnRenamed("labeltmp", "label")

    df = df.drop("_c0")

    df
  }

  // preprocessing for training data
  def prep_train(df_train: DataFrame): DataFrame = {

    var df = prep(df_train)

    val word2VecModel = new TrainWord2Vec(df, vec_size, min_count).word2VecModel

    df = word2VecModel.transform(df)

    df = df.drop("Text", "Tokens", "Filtered")

    df = df.withColumnRenamed("Sentiment", "label")
    df = df.withColumnRenamed("Vec", "features")
    df = df.withColumn("labeltmp", df.col("label").cast(FloatType))
      .drop("label")
      .withColumnRenamed("labeltmp", "label")

    df = df.drop("_c0")

    df
  }
}
