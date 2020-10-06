import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.types.FloatType

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
    var df_train = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/train_data.csv")

    var df_test = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/test_data.csv")

    val preprocessing = new Preprocessing(spark)

    df_train = preprocessing.prep_train(df_train)
    df_test = preprocessing.prep_test(df_test)

    println("Training")
    df_train.show(20, false)

    println("Testing")
    df_test.show(20, false)

    var logisticRegression = new LogisticRegression()
      .setMaxIter(1000)
      .setRegParam(0.1)

    val logisticRegressionModel = logisticRegression.fit(df_train)
    val predicitons = logisticRegressionModel.transform(df_test)

    println("Logistric Regression")
    predicitons.show(20, false)

    // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
    // example
    val trainingSummary = logisticRegressionModel.binarySummary

    println("History:")
    // Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))


    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predicitons)
    print("Accuracy: ")
    println(accuracy)

    logisticRegressionModel.save("logRegModel")

  }
}
