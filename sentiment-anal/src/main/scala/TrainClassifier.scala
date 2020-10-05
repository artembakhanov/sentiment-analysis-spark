import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace

object TrainClassifier {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    // set up environment
    val conf = new SparkConf()
      .setAppName("Training")
    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder()
      .appName("Training")
      .config(conf)
      .getOrCreate()

    // reading the training data and storing it in dataframe
    var df = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/train_data.csv")

    var df_test = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/test_data.csv")

    val preprocessing = new Preprocessing(spark)

    df = preprocessing.prep_train(df)
    df_test = preprocessing.prep_test(df_test)

    println("Training")
    df.show(20, false)

    print("Testing")
    df_test.show(20, false)


  }
}
