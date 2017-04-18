package com.ferenc.kis.poc.spark.ml

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object RandomForest {

  val INPUT_PATH = "src/main/resources/data/bank.csv"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName("random-forest").getOrCreate()

    val df = spark.read
      .option("header", true)
      .option("delimiter", ";")
      .csv(INPUT_PATH)

    import df.sparkSession.implicits._
    val raw = df.select(
      when($"y" === "no", 1.0).otherwise(0.0).as("outcome"),
      $"age".cast(DoubleType).as("age"),
      when($"marital" === "single", 1.0).otherwise(0.0).as("single"),
      when($"marital" === "married", 1.0).otherwise(0.0).as("married"),
      when($"marital" === "divorced", 1.0).otherwise(0.0).as("divorced"),
      when($"education" === "primary", 1.0).otherwise(0.0).as("primary"),
      when($"education" === "secondary", 1.0).otherwise(0.0).as("secondary"),
      when($"education" === "tertiary", 1.0).otherwise(0.0).as("tertiary"),
      when($"default" === "no", 1.0).otherwise(0.0).as("default"),
      when($"balance" === "no", 1.0).otherwise(0.0).as("balance"),
      when($"loan" === "no", 1.0).otherwise(0.0).as("loan")
    )

    raw.printSchema()
    raw.show()

    val OUTCOME_FIELD = "outcome"
    val FEATURE_FIELDS = List("age", "single", "married", "divorced", "primary", "secondary", "tertiary",
      "default", "balance", "loan")
    FEATURE_FIELDS.foreach(field => println(raw.stat.corr(OUTCOME_FIELD, field)))

    val featureColumns = FEATURE_FIELDS.map(raw(_))
    val denseVector = udf( (features: Seq[Double]) => Vectors.dense(features.toArray) )
    val mlReadyData = raw.select($"$OUTCOME_FIELD".as("label"), denseVector(array(featureColumns : _*)).as("features"))

    mlReadyData.show()

    val pca = new PCA().setK(3).setInputCol("features").setOutputCol("pcaFeatures")
    val pcaModel = pca.fit(mlReadyData)
    println(pcaModel.explainedVariance.values.mkString(","))
    val pcaData = pcaModel.transform(mlReadyData)

    pcaData.printSchema()
    pcaData.show()

    val Array(trainingData, testData) = pcaData.randomSplit(Array(0.7, 0.3))
    println(trainingData.count())
    println(testData.count())

    val rmfClassifier = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("pcaFeatures")
      .setMaxDepth(5).setNumTrees(5)
    val rmfModel = rmfClassifier.fit(trainingData)

    println(rmfModel.toDebugString)

    val rmfPredictions = rmfModel.transform(testData)
    rmfPredictions.show()

    val evaluator = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
      .setLabelCol("label").setMetricName("accuracy")

    println(evaluator.evaluate(rmfPredictions))

    // confusion matrix
    rmfPredictions.groupBy("label", "prediction").count().show()
  }
}
