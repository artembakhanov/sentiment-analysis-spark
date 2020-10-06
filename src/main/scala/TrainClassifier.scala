import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

object TrainClassifier {

  // function for training Logistic Regression
  def train_logReg(df_train: DataFrame) = {
    // init
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

    // Note: Below is the grid search from which the best parameters were achieved

    // grid search
    // https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/classification/LogisticRegression.html
    /*val paramGrid = new ParamGridBuilder()
      .addGrid(logisticRegression.elasticNetParam, Array(0, 0.5, 0.8, 1))
      .addGrid(logisticRegression.fitIntercept, Array(true, false))
      .addGrid(logisticRegression.maxIter, Array(1000))
      .addGrid(logisticRegression.regParam, Array(0, 0.1, 0.2))
      .addGrid(logisticRegression.threshold, Array(0.5))
      .build()*/

    /*
      Best Params:
      {
        logreg_0a99d82c11df-aggregationDepth: 2,
        logreg_0a99d82c11df-elasticNetParam: 0.5,
        logreg_0a99d82c11df-family: auto,
        logreg_0a99d82c11df-featuresCol: features,
        logreg_0a99d82c11df-fitIntercept: true,
        logreg_0a99d82c11df-labelCol: label,
        logreg_0a99d82c11df-maxIter: 1000,
        logreg_0a99d82c11df-predictionCol: prediction,
        logreg_0a99d82c11df-probabilityCol: probability,
        logreg_0a99d82c11df-rawPredictionCol: rawPrediction,
        logreg_0a99d82c11df-regParam: 0.0,
        logreg_0a99d82c11df-standardization: true,
        logreg_0a99d82c11df-threshold: 0.5,
        logreg_0a99d82c11df-tol: 1.0E-6
      }
      Accuracy: 0.753
    */

    /*// cross validation
    val crossval = new CrossValidator()
      .setEstimator(logisticRegression)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = crossval.fit(df_train)

    println("Logistic Regression - Best Params")
    println(cvModel.bestModel.extractParamMap())

    cvModel.save("logRegModel")

    cvModel*/
  }

  // function for training Random Forest
  def train_RandForest(df_train: DataFrame) = {
    //init
    val randomForestClassifier = new RandomForestClassifier()
      .setMaxDepth(7)
      .setNumTrees(40)

    //train
    val randomForestClassificationModel = randomForestClassifier.fit(df_train)

    randomForestClassificationModel.save("randomForestModel")

    randomForestClassificationModel

    // Note: Below is the grid search from which the best parameters were achieved

    // grid search
    // https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/classification/RandomForestClassifier.html
    /*val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .addGrid(randomForestClassifier.maxDepth, Array(3, 5, 7))
      .addGrid(randomForestClassifier.numTrees, Array(20, 40))
      .build()*/

    /*
      Best Params:
      {
        rfc_ba215569256d-bootstrap: true,
        rfc_ba215569256d-cacheNodeIds: false,
        rfc_ba215569256d-checkpointInterval: 10,
        rfc_ba215569256d-featureSubsetStrategy: auto,
        rfc_ba215569256d-featuresCol: features,
        rfc_ba215569256d-impurity: gini,
        rfc_ba215569256d-labelCol: label,
        rfc_ba215569256d-leafCol: ,
        rfc_ba215569256d-maxBins: 32,
        rfc_ba215569256d-maxDepth: 7,
        rfc_ba215569256d-maxMemoryInMB: 256,
        rfc_ba215569256d-minInfoGain: 0.0,
        rfc_ba215569256d-minInstancesPerNode: 1,
        rfc_ba215569256d-minWeightFractionPerNode: 0.0,
        rfc_ba215569256d-numTrees: 40,
        rfc_ba215569256d-predictionCol: prediction,
        rfc_ba215569256d-probabilityCol: probability,
        rfc_ba215569256d-rawPredictionCol: rawPrediction,
        rfc_ba215569256d-seed: 207336481,
        rfc_ba215569256d-subsamplingRate: 1.0
      }
      Accuracy: 0.7524448775963525
    */

    // cross validation
    /*val crossval = new CrossValidator()
      .setEstimator(randomForestClassifier)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = crossval.fit(df_train)

    var logisticRegressionModel = cvModel.bestModel
    println("Random Forest - Best Params")
    println(cvModel.bestModel.extractParamMap())

    cvModel.save("randomForestModel")

    cvModel*/
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

