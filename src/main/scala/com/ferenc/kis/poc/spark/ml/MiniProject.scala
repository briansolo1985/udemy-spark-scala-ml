package com.ferenc.kis.poc.spark.ml

import java.lang.Math

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, StandardScaler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object MiniProject {

  val SOURCE_PATH = "src/main/resources/data/credit-card-default-1000.csv"

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[4]").appName("linear-regression").getOrCreate()

   import spark.implicits._

    val schema = StructType(
      Seq(
        StructField("CUSTID", IntegerType, false),
        StructField("LIMIT_BAL", DoubleType, false),
        StructField("SEX", StringType, false),
        StructField("EDUCATION", DoubleType, false),
        StructField("MARRIAGE", DoubleType, false),
        StructField("AGE", DoubleType, false),
        StructField("PAY_1", DoubleType, false),
        StructField("PAY_2", DoubleType, false),
        StructField("PAY_3", DoubleType, false),
        StructField("PAY_4", DoubleType, false),
        StructField("PAY_5", DoubleType, false),
        StructField("PAY_6", DoubleType, false),
        StructField("BILL_AMT1", DoubleType, false),
        StructField("BILL_AMT2", DoubleType, false),
        StructField("BILL_AMT3", DoubleType, false),
        StructField("BILL_AMT4", DoubleType, false),
        StructField("BILL_AMT5", DoubleType, false),
        StructField("BILL_AMT6", DoubleType, false),
        StructField("PAY_AMT1", DoubleType, false),
        StructField("PAY_AMT2", DoubleType, false),
        StructField("PAY_AMT3", DoubleType, false),
        StructField("PAY_AMT4", DoubleType, false),
        StructField("PAY_AMT5", DoubleType, false),
        StructField("PAY_AMT6", DoubleType, false),
        StructField("DEFAULTED", DoubleType, false)
      )
    )

    // PR#01 - read with cleanse
    val cleansed = spark.read
      .option("header", true)
      .option("mode", "DROPMALFORMED")
      .schema(schema)
      .csv(SOURCE_PATH)
//    cleansed.show(10000)

    // PR#02 - defaults correlation with SEX
    val custidWithNumericFeatures = cleansed.withColumnRenamed("SEX", "OLD_SEX")
      .withColumn("SEX", when($"OLD_SEX" === "M", 1.0).when($"OLD_SEX" === "F", 2.0).otherwise($"OLD_SEX".cast(DoubleType)))
      .drop($"OLD_SEX")
//    custidWithNumericFeatures.show()

    val defaultsCorrelationWithSex = custidWithNumericFeatures.groupBy(when($"SEX" === 1.0, "Male").otherwise("Female").as("SEX_NAME"))
      .agg(count("*").as("TOTAL"), sum("DEFAULTED").as("DEFAULTS"))
      .withColumn("PER_DEFAULT", round($"DEFAULTS" * 100 / $"TOTAL"))
//    defaultsCorrelationWithSex.show()

    // PR#03 - defaults correlation with Marital status and Education
    val defaultsCorrelationWithMaritalStatusAndEducation = custidWithNumericFeatures.groupBy(
        when($"MARRIAGE" === 1.0, "Single").when($"MARRIAGE" === 2.0, "Married").otherwise("Others").as("MARRIAGE_NAME"),
        when($"EDUCATION" === 1.0, "Graduate").when($"EDUCATION" === 2.0, "University").when($"EDUCATION" === 3.0, "High School")
          .otherwise("Others").as("EDUCATION_NAME")
      )
      .agg(count("*").as("TOTAL"), sum("DEFAULTED").as("DEFAULTS"))
      .withColumn("PER_DEFAULT", round($"DEFAULTS" * 100 / $"TOTAL"))
//    defaultsCorrelationWithMaritalStatusAndEducation.show()

    // PR#04 - defaults correlation with average payment duration
    val defaultsCorrelationWithAveragePaymentDuration = custidWithNumericFeatures.groupBy(
        bround((abs($"PAY_1") + abs($"PAY_2") + abs($"PAY_3") + abs($"PAY_4") + abs($"PAY_5") + abs($"PAY_6")) / 6).as("AVG_PAY_DUR")
      )
      .agg(count("*").as("TOTAL"), sum("DEFAULTED").as("DEFAULTS"))
      .withColumn("PER_DEFAULT", round($"DEFAULTS" * 100 / $"TOTAL"))
//    defaultsCorrelationWithAveragePaymentDuration.show()

    // PR#04 - defaults prediction models

    // correlation analysis
    val features = custidWithNumericFeatures.drop("CUSTID")
//    features.show(10000)
//    features.printSchema()
//    features.columns.foreach(c1 => {
//      features.columns.foreach(c2 => {
//        if(c1 != c2) {
//          println(s"Correlation between $c1 and $c2 is ${features.stat.corr(c1, c2)}")
//        }
//      })
//    })

    val featureColumnNames = List("LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4",
      "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
      "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "SEX")
    val featureColumns = featureColumnNames.map(name => col(name))

    val denseVector = udf( (features: Seq[Double]) => Vectors.dense(features.toArray) )
    val mlData = features.select($"DEFAULTED" as "label", denseVector(array(featureColumns : _*)) as "features")
//    mlData.show()

    val Array(decisionTreeTrainingData, decisionTreeTestData) = mlData.randomSplit(Array(0.7, 0.3))

    // DECISION TREE
    val decisionTreeClassifier = new DecisionTreeClassifier().setMaxDepth(6)
    val model = decisionTreeClassifier.fit(decisionTreeTrainingData)

    val decisionTreePredictions = model.transform(decisionTreeTestData)
    decisionTreePredictions.show()

    evaluate(decisionTreePredictions)
    confusionMatrix(decisionTreePredictions)

    // RANDOM FOREST
    val rmfClassifier = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
      .setMaxDepth(5).setNumTrees(5)
    val rmfModel = rmfClassifier.fit(decisionTreeTrainingData)

//    println(rmfModel.toDebugString)

    val rmfPredictions = rmfModel.transform(decisionTreeTestData)
//    rmfPredictions.show()

    evaluate(rmfPredictions)
    confusionMatrix(rmfPredictions)

    // NAIVE BAYES
    val positiveDenseVector = udf( (features: Seq[Double]) => Vectors.dense(features.map(d => Math.abs(d)).toArray) )
    val naiveBayesData = features.select($"DEFAULTED" as "label", positiveDenseVector(array(featureColumns : _*)) as "features")
    //    naiveBayesData.show()

    val Array(nbTrainingData, nbTestData) = naiveBayesData.randomSplit(Array(0.7, 0.3))

    val nbClassifier = new NaiveBayes().setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction")

    val nbModel = nbClassifier.fit(nbTrainingData)

    val nbPredictions = nbModel.transform(nbTestData)
    nbPredictions.show()

    evaluate(nbPredictions)
    confusionMatrix(nbPredictions)

    // PR#06 - defaults prediction models
    val kMeansPrepared = features.select(
      lit(1.0).as("label"),
      denseVector(array($"SEX", $"EDUCATION", $"MARRIAGE", bround($"AGE", -1).as("AGE_RANGE"))) as "features"
    )

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)

    val scalerModel = scaler.fit(kMeansPrepared)
    val kMeansScaledData = scalerModel.transform(kMeansPrepared)
    kMeansScaledData.show()

    val kMeans = new KMeans().setK(4).setSeed(1L)

    val kMeansModel = kMeans.fit(kMeansScaledData)
    val kMeansPredictions = kMeansModel.transform(kMeansScaledData)

    kMeansPredictions.show()
  }

  def evaluate(predictions: DataFrame) = {
    val evaluator = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
      .setLabelCol("label").setMetricName("accuracy")
    println(s"Accuracy=${evaluator.evaluate(predictions)}")
  }

  def confusionMatrix(predictions: DataFrame) = {
    println("Confusion matrix")
    predictions.groupBy("label", "prediction").count().show()
  }
}
