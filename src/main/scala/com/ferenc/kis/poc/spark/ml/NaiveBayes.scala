package com.ferenc.kis.poc.spark.ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.types.{DoubleType, StringType}

object NaiveBayes {
  val INPUT_PATH = "d:\\Projects\\SparkML\\data\\SMSSpamCollection.csv"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName("naive-bayes").getOrCreate()

    val df = spark.read
      .option("header", false)
      .csv(INPUT_PATH)

    df.show()

    import df.sparkSession.implicits._
    val smsVectors = df.select(
      when($"_c0" === "spam", 1.0).otherwise(0.0).cast(DoubleType).as("label"),
      $"_c1".cast(StringType).as("message")
    )

    smsVectors.show()

    val Array(trainingData, testData) = smsVectors.randomSplit(Array(0.7, 0.3))

    // Pipeline
    val tokenizer = new Tokenizer().setInputCol("message").setOutputCol("words")
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("tempFeatures")
    val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
    val nbClassifier = new NaiveBayes().setLabelCol("label").setFeaturesCol(idf.getOutputCol).setPredictionCol("prediction")

    val pipeLine = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, nbClassifier))

    val nbModel = pipeLine.fit(trainingData)

    val predictions = nbModel.transform(testData)
    predictions.show()

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
      .setPredictionCol("prediction").setMetricName("accuracy")
    println($"Accuracy=${evaluator.evaluate(predictions)})")

    predictions.groupBy($"label", $"prediction").count().show()
  }
}
