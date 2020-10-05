import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.{SparkConf, SparkContext}

object Preprocessing {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    // set up environment
    val conf = new SparkConf()
      .setAppName("Preprocessing")
    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder()
      .appName("Preprocessing")
      .config(conf)
      .getOrCreate()

    // reading the training data and storing it in dataframe
    var df = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/train_data.csv")

    // removing usernames
    df = df.withColumn("Text", regexp_replace(df.col("Text"), "@\\w+", " "))

    // defining tokenizer
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("Text")
      .setOutputCol("Words")
      .setPattern("\\w+")
      .setGaps(false)

    // tokenizing the text
    val tokenized = regexTokenizer.transform(df)

    // writing to file
    tokenized.coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("tokenized.csv")
  }
}
