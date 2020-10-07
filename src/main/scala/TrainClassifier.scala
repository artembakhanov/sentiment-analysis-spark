import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

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
    logisticRegressionModel.save("logRegModel")

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

  def train_SVC(df_train: DataFrame) = {
    // init
    // Note: the best parameters were achieved in the grid search (check out Classifiers.scala)
    val svc = new LinearSVC()
      .setAggregationDepth(2)
      .setMaxIter(150)
      .setThreshold(0.6)
      .setRegParam(0.0)

    //train
    val svcModel = svc.fit(df_train)

    // save
    svcModel.save("svcModel")

    svcModel
  }

  // function for calculating f1 score
  def calc_f1Score(sc: SparkContext, predictions: DataFrame) = {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // rdd storing tuples in the form of (probability, label)
    val predictionsAndLabels = predictions.map(x => (x.getAs[Double]("prediction"), x.getAs[Double]("label"))).rdd


    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

    // calculating f1 score
    val f1Score = metrics.fMeasureByThreshold
      .collect()

    // taking max value of f1 score
    f1Score.maxBy(_._1)._2 // f1 score
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
    val preprocessing = new Preprocessing(sc, spark, vec_size = 25, min_count = 15)
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
    println(s"F1 score: ${calc_f1Score(sc, predictions1)}")



    // training Random Forest Model
    val randomForestModel = train_RandForest(df_train)
    // testing the model
    val predictions2 = randomForestModel.transform(df_test)
    // printing the results
    println("Random Forest")
    predictions2.show(20, true)
    // evaluating the model
    println(s"F1 score: ${calc_f1Score(sc, predictions2)}")


    // training Random Forest Model
    val svcModel = train_SVC(df_train)
    // testing the model
    val predictions3 = svcModel.transform(df_test)
    // printing the results
    println("SVC")
    predictions3.show(20, true)
    // evaluating the model
    println(s"F1 score: ${calc_f1Score(sc, predictions3)}")


  }
}

