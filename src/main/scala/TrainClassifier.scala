import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession, SQLContext}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object TrainClassifier {

  // function for training Logistic Regression
  def train_logReg(df_train: DataFrame) = {
    // init
    // Note: the best parameters were achieved in the grid search (check out Classifiers.scala)
    val logisticRegression = new LogisticRegression()
      .setElasticNetParam(0.5)
      .setFitIntercept(true)
      .setMaxIter(1000)
      .setRegParam(0)
      .setThreshold(0.5)

    // train
    val logisticRegressionModel = logisticRegression.fit(df_train)

    // save
    logisticRegressionModel.save("logRegModelTEST")

    logisticRegressionModel
  }

  // function for training Random Forest
  def train_RandForest(df_train: DataFrame) = {
    // init
    // Note: the best parameters were achieved in the grid search (check out Classifiers.scala)
    val randomForestClassifier = new RandomForestClassifier()
      .setMaxDepth(7)
      .setNumTrees(40)

    //train
    val randomForestClassificationModel = randomForestClassifier.fit(df_train)

    // save
    randomForestClassificationModel.save("randomForestModelTEST")

    randomForestClassificationModel
  }

  // function for calculating f1 score
  def calc_f1Score(sc: SparkContext, predictions: DataFrame): Tuple2[Double, Double] = {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // rdd storing tuples in the form of (probability, label)
    val predictionsAndLabels = predictions.map(x => (x.getAs[Vector]("probability")(1), x.getAs[Double]("label"))).rdd

    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

    // calculating f1 score
    val f1Score = metrics.fMeasureByThreshold
      .collect()

    // taking max value of f1 score
    f1Score.maxBy(_._2) //Tuple2(threshold, f1 score)
  }


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

    // reading the training and testing data and storing it in dataframe
    var df_train = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/train_data.csv")

    var df_test = spark.read.format("csv")
      .option("header", "true")
      .load("/user/tesem/test_data.csv")

    // preprocessing the data
    val preprocessing = new Preprocessing(spark, vec_size = 25, min_count = 15)
    df_train = preprocessing.prep_train(df_train)
    df_test = preprocessing.prep_test(df_test)

    // printing dataframes after preprocessing
    println("Training")
    df_train.show(20, true)
    println("Testing")
    df_test.show(20, true)

    // training Logistic Regression Model
    val logisticRegressionModel = train_logReg(df_train)
    // testing the model
    var predictions1 = logisticRegressionModel.transform(df_test)
    // printing the results
    println("Logistic Regression")
    predictions1.show(20, true)

    // evaluating the model
    val evaluator1 = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy1 = evaluator1.evaluate(predictions1)
    print("Accuracy: ")
    println(accuracy1)
    println(s"The best f1 score: ${calc_f1Score(sc, predictions1)}")

    // training Random Forest Model
    val randomForestModel = train_RandForest(df_train)
    // testing the model
    val predictions2 = randomForestModel.transform(df_test)
    // printing the results
    println("Random Forest")
    predictions2.show(20, true)


    // evaluating the model
    val evaluator2 = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy2 = evaluator2.evaluate(predictions2)
    print("Accuracy: ")
    println(accuracy2)
    println(s"The best f1 score: ${calc_f1Score(sc, predictions2)}")

  }
}

