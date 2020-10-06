import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

object TrainClassifier {

  // function for training Logistic Regression
  def train_logReg(df_train: DataFrame, maxIter: Int, regParam: Double): LogisticRegressionModel = {
    // init
    val logisticRegression = new LogisticRegression()
      .setMaxIter(maxIter)
      .setRegParam(regParam)

    // train
    val logisticRegressionModel = logisticRegression.fit(df_train)
    // save
    logisticRegressionModel.save("logRegModel")

    logisticRegressionModel
  }

  // function for training Random Forest
  def train_RandForest(df_train: DataFrame): RandomForestClassificationModel = {
    //init
    val randomForestClassifier = new RandomForestClassifier()

    //train
    val randomForestClassificationModel = randomForestClassifier.fit(df_train)

    randomForestClassificationModel.save("randomForestModel")

    randomForestClassificationModel
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
    val logisticRegressionModel = train_logReg(df_train, 1000, 0.1)
    // testing the model
    var predictions1 = logisticRegressionModel.transform(df_test)
    // printing the results
    println("Logistic Regression")
    predictions1.show(20, true)

    // Extract the summary from the returned LogisticRegressionModel
    val trainingSummary1 = logisticRegressionModel.binarySummary
    println("History:")
    // Obtain the objective per iteration.
    val objectiveHistory1 = trainingSummary1.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory1.foreach(loss => println(loss))

    // evaluating the model
    val evaluator1 = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy1 = evaluator1.evaluate(predictions1)
    print("Accuracy: ")
    println(accuracy1)


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

  }
}
