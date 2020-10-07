import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.{Classifier, LinearSVC, LinearSVCModel, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap

class Classifiers {

  def crossValidator(classifier: Classifier[_, _, _], paramGrid: Array[ParamMap], numFolds: Int = 5): CrossValidator = {
    val crossval = new CrossValidator()
      .setEstimator(classifier)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    crossval
  }

  def trainSaveClassifier(crossValidator: CrossValidator, df_train: DataFrame, modelName: String, saveTo: String) = {
    val cvModel = crossValidator.fit(df_train)

    println(f"$modelName%s - Best Params")
    println(cvModel.bestModel.extractParamMap())
    cvModel.save(saveTo)

    cvModel
  }

  def logisticRegression(df_train: DataFrame) = {
    val logisticRegression = new LogisticRegression()

    // grid search
    // https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/classification/LogisticRegression.html
    val paramGrid = new ParamGridBuilder()
      .addGrid(logisticRegression.elasticNetParam, Array(0, 0.5, 0.8, 1))
      .addGrid(logisticRegression.fitIntercept, Array(true, false))
      .addGrid(logisticRegression.maxIter, Array(1000))
      .addGrid(logisticRegression.regParam, Array(0, 0.1, 0.2))
      .addGrid(logisticRegression.threshold, Array(0.5))
      .build()

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

    // cross validation
    val crossval = crossValidator(logisticRegression, paramGrid, 5)
    trainSaveClassifier(crossval, df_train, "Logistic Regression", "logRegModel")
  }

  def randomForest(df_train: DataFrame) = {
    val randomForestClassifier = new RandomForestClassifier()

    // grid search
    // https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/classification/RandomForestClassifier.html
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .addGrid(randomForestClassifier.maxDepth, Array(3, 5, 7))
      .addGrid(randomForestClassifier.numTrees, Array(20, 40))
      .build()

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
    val crossval = crossValidator(randomForestClassifier, paramGrid, 5)
    trainSaveClassifier(crossval, df_train, "Random Forest", "randomForestModel")
  }


  def svc(df_train: DataFrame): Unit = {
    val svcModel = new LinearSVC()

    val paramGrid = new ParamGridBuilder()
      .addGrid(svcModel.aggregationDepth, Array(2, 3))
      .addGrid(svcModel.maxIter, Array(100, 150))
      .addGrid(svcModel.threshold, Array(0.4, 0.5, 0.6))
      .addGrid(svcModel.regParam, Array(0, 0.1, 0.2))
      .build()


    val crossval = crossValidator(svcModel, paramGrid, 5)
    trainSaveClassifier(crossval, df_train, "Linear SVC", "svcModel")
  }

}
