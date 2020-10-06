import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.DataFrame

import scala.reflect.io.Directory

class TrainWord2Vec(df_train: DataFrame, vec_size: Int, min_count: Int) {

  val word2Vec = new Word2Vec()
    .setInputCol("Filtered")
    .setOutputCol("Vec")
    .setVectorSize(vec_size)
    .setMinCount(min_count)

  val word2VecModel = word2Vec.fit(df_train)

  val directory = new Directory(new File("hdfs:///word2VecModel"))
  directory.deleteRecursively()

  word2VecModel.save("word2VecModel")
}
