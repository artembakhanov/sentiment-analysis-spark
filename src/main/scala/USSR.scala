import org.apache.spark.ml.classification.{ClassificationModel, LinearSVCModel, LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object USSR {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("USSRing")

    // unites files to one
    val spark2 = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    var models = Array[(ClassificationModel[_, _], String)]()

    if (args.contains("logreg")) {
      models = models :+ (LogisticRegressionModel.load("logRegModel"), "logRegModel")
      println("Adding logRegModel")
    }
    if (args.contains("randforest")) {
      models = models :+ (RandomForestClassificationModel.load("randomForestModel"), "randomForestModel")
      println("Adding randomForestModel")
    }

    if (args.contains("svc")) {
      models = models :+ (LinearSVCModel.load("svcModel"), "svcModel")
      println("Adding svcModel")
    }

    for ((_, name) <- models) {
      spark2.read.csv(f"stream/$name%s/").coalesce(1).write.csv(f"stream_final/$name%s/")
    }
  }
}
