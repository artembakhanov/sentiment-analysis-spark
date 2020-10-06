import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}


object StreamProcesser {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName")
    val sc = new SparkContext(conf)


    import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
    import org.apache.spark.sql.{Row, SQLContext, SparkSession}
    import org.apache.spark.streaming.{Seconds, StreamingContext, Time}
    import org.apache.spark.{SparkConf, SparkContext}
    val spark = SparkSession.builder().appName("Training").config(sc.getConf).getOrCreate()
    val ssc = new StreamingContext(sc, Seconds(2))

    val ds = ssc.socketTextStream("10.90.138.32", 8989)

    val schema = new StructType().add(StructField("Sentiment", IntegerType, true)).add(StructField("Text", StringType, false))
    ds.foreachRDD(
      tweet => {
        val stream = spark.createDataFrame(tweet.map(attributes => Row(1, attributes)), schema)
        stream.show()
      }
    )
    ssc.start()
    ssc.awaitTerminationOrTimeout(60 * 1000)
    ssc.stop()
  }
}
