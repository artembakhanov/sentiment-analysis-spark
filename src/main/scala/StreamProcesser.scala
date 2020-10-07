import java.security.MessageDigest

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
    val ssc = new StreamingContext(sc, Seconds(60))

    val ds = ssc.socketTextStream("10.90.138.32", 8989)

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
    //    val models = Array((LogisticRegressionModel.load("logRegModel"), "logRegModel"), (RandomForestClassificationModel.load("randomForestModel"), "randomForestModel"), (LinearSVCModel.load("svcModel"), "svcModel"))

    val schema = new StructType()
      .add(StructField("Sentiment", IntegerType, true))
      .add(StructField("Text", StringType, false))
      .add(StructField("OriginalText", StringType, false))

    ds.foreachRDD(
      tweet => {
        val stream = spark.createDataFrame(tweet.map(attributes => Row(1, attributes, attributes)), schema)

        if (!stream.isEmpty) {
          val preprocessing = new Preprocessing(sc, spark, 20, 20)
          val df_test = preprocessing.prep_test(stream)

          for ((model, name) <- models) {
            val predictions = model.transform(df_test)
            println(name)
            predictions
              .withColumn("Timestamp", current_timestamp())
              .select("Timestamp", "OriginalText", "prediction")
              .coalesce(1)
              .write.mode(SaveMode.Append).csv(f"stream/$name%s/")
          }
        }
      }
    )
    ssc.start()
    ssc.awaitTerminationOrTimeout(taskDuration)
    ssc.stop()

    val spark2 = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    spark2.read.csv("stream/logRegModel/").coalesce(1).write.csv("stream_final/logRegModel/")
    spark2.read.csv("stream/randomForestModel").coalesce(1).write.csv("stream_final/randomForestModel/")
    spark2.read.csv("stream/svcModel").coalesce(1).write.csv("stream_final/svcModel/")

  }
}
