import org.apache.spark.ml.classification.{ClassificationModel, LinearSVCModel, LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.functions.current_timestamp
import org.apache.spark.sql.types.{DateType, IntegerType, StringType, StructField, StructType, TimestampType}
import org.apache.spark.sql.{Row, SQLContext, SaveMode, SparkSession}
import org.apache.spark.streaming.{Seconds, StreamingContext, Time}
import org.apache.spark.{SparkConf, SparkContext}

import scala.:+
import scala.concurrent.duration.Duration


object StreamProcesser {
  def main(args: Array[String]): Unit = {

    var taskDuration = 120 * 1000L // 2 minutes
    if (args.length > 0) {
      taskDuration = Duration(args(0)).toMillis
    }


    val conf = new SparkConf().setAppName("Streaming")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().appName("Streaming").config(sc.getConf).getOrCreate()
    val ssc = new StreamingContext(sc, Seconds(2))

    val ds = ssc.socketTextStream("10.90.138.32", 8989)

    val models = Array[(ClassificationModel[_, _], String)]()
    if (args.contains("logreg"))
      models :+ (LogisticRegressionModel.load("logRegModel"), "logRegModel")
    if (args.contains("randforest"))
      models :+ (RandomForestClassificationModel.load("randomForestModel"), "randomForestModel")
    if (args.contains("svc"))
      models :+ (LinearSVCModel.load("svcModel"), "svcModel")
//    val models = Array((LogisticRegressionModel.load("logRegModel"), "logRegModel"), (RandomForestClassificationModel.load("randomForestModel"), "randomForestModel"), (LinearSVCModel.load("svcModel"), "svcModel"))
    val schema = new StructType()
      .add(StructField("Sentiment", IntegerType, true))
      .add(StructField("Text", StringType, false))
      .add(StructField("OriginalText", StringType, false))

    ds.foreachRDD(
      tweet => {
        val stream = spark.createDataFrame(tweet.map(attributes => Row(1, attributes, attributes)), schema)

        if (!stream.isEmpty) {
          val preprocessing = new Preprocessing(spark, 20, 20)
          val df_test = preprocessing.prep_test(stream)

          for ((model, name) <- models) {
            val predictions = model.transform(df_test)
            predictions
              .withColumn("Timestamp", current_timestamp())
              .select("Timestamp", "OriginalText", "prediction")
              .write.mode(SaveMode.Append).csv(f"stream/$name%s/")
          }
        }
      }
    )
    ssc.start()
    ssc.awaitTerminationOrTimeout(taskDuration)
    ssc.stop()
  }
}
