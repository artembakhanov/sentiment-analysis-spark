import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.DataFrame

// training word2vec on df_train
class TrainWord2Vec(df_train: DataFrame, vec_size: Int, min_count: Int) {

  // init
  val word2Vec = new Word2Vec()
    .setInputCol("Filtered")
    .setOutputCol("Vec")
    .setVectorSize(vec_size)
    .setMinCount(min_count)

  // fitting the data
  val word2VecModel = word2Vec.fit(df_train)

  // saving the model to hdfs for future use
  word2VecModel.save("word2VecModel")
}
