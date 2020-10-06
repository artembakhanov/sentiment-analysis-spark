import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext, SaveMode, SparkSession}
import org.apache.spark.streaming.{Seconds, StreamingContext, Time}
import org.apache.spark.{SparkConf, SparkContext}


object StreamProcesser {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Streaming")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().appName("Streaming").config(sc.getConf).getOrCreate()
    val ssc = new StreamingContext(sc, Seconds(2))

    val ds = ssc.socketTextStream("10.90.138.32", 8989)

    val logisticRegressionModel = LogisticRegressionModel.load("logRegModel")
    val schema = new StructType()
      .add(StructField("Sentiment", IntegerType, true))
      .add(StructField("Text", StringType, false))

    ds.foreachRDD(
      tweet => {
        val stream = spark.createDataFrame(tweet.map(attributes => Row(1, attributes)), schema)

        if (!stream.isEmpty) {
          val preprocessing = new Preprocessing(spark, 20, 20)
          val df_test = preprocessing.prep_test(stream)

          val predictions = logisticRegressionModel.transform(df_test)
          predictions.write.mode(SaveMode.Append).csv("stream/pred.csv")
        }

      }
    )
    ssc.start()
    ssc.awaitTerminationOrTimeout(120 * 1000)
    ssc.stop()
  }
}
