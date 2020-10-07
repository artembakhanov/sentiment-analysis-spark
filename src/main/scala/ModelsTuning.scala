import java.security.MessageDigest

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

object ModelsTuning {
  val models = Map(
    "logreg" -> 1,
    "randforest" -> 1,
    "svc" -> 1
  )

  def tuneModel(model: String, df_train: DataFrame): Unit = {
    val classifiers = new Classifiers()
    if (model == "logreg") {
      classifiers.logisticRegression(df_train)
    } else if (model == "randforest") {
      classifiers.randomForest(df_train)
    } else if (model == "svc") {
      classifiers.svc(df_train)
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length < 1 || !models.contains(args(0))) {
      println("Please, specify the model to tune: logreg, randforest, svc")
      sys.exit(1)
    }

    // set up environment
    // we use hashing against spies ;)
    val conf = new SparkConf()
      .setAppName(f"Model Tuning - ${MessageDigest.getInstance("MD5").digest(args(0).getBytes).map(_.toChar).mkString}%s")

    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder()
      .appName(f"Model Tuning - ${MessageDigest.getInstance("MD5").digest(args(0).getBytes).map(_.toChar).mkString}%s")
      .config(conf)
      .getOrCreate()

    var df_train = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/train_data.csv")

    // preprocessing the data
    val preprocessing = new Preprocessing(sc, spark, vec_size = 25, min_count = 15)
    df_train = preprocessing.prep_train(df_train)

    tuneModel(args(0), df_train)
  }
}
