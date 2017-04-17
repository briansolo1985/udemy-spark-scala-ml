package com.ferenc.kis.poc.spark.ml

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object LinerRegression {

  val MILES_PER_GALLON_INPUT_PATH = "d:\\Projects\\SparkML\\data\\auto-miles-per-gallon.csv"
  val NULL_VALUE = "?"

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[4]").appName("linear-regression").getOrCreate()

    val schema = StructType(
      Seq(
        StructField("MPG", DoubleType, false),
        StructField("DISPLACEMENT", DoubleType, false),
        StructField("CYLINDERS", DoubleType, false),
        StructField("HP", DoubleType, false),
        StructField("WEIGHT", DoubleType, false),
        StructField("ACCELERATION", DoubleType, false),
        StructField("MODELYEAR", DoubleType, false),
        StructField("NAME", StringType, false)
      )
    )

    import spark.implicits._

    val df = spark.read
      .option("header", true)
      .option("nullValue", NULL_VALUE)
      .schema(schema)
      .csv(MILES_PER_GALLON_INPUT_PATH)

    val cleansed = df.select($"MPG", $"CYLINDERS", when($"HP".isNotNull, $"HP").otherwise(80.0).as("HP"), $"ACCELERATION", $"MODELYEAR").cache()

    cleansed.show()
    println(cleansed.count())

    val columnNames = List("MPG", "CYLINDERS", "HP", "ACCELERATION", "MODELYEAR")

    // You can plot correlation between variables in Zeppelin with SparkSQL
    columnNames.foreach(c => {
      println(s"Correlation between ${columnNames.head} and $c is ${cleansed.stat.corr(columnNames.head, c)}")
    })

    val denseVector = udf( (features: Seq[Double]) => Vectors.dense(features.toArray) )
    val mlData = cleansed.select($"MPG" as "label", denseVector(array($"CYLINDERS", $"HP", $"ACCELERATION", $"MODELYEAR")) as "features")
    mlData.show()

    val Array(trainingData, testData) = mlData.randomSplit(Array(0.8, 0.2))
    println(trainingData.count())
    println(testData.count())

    val lr = new LinearRegression().setMaxIter(25)
    val model: LinearRegressionModel = lr.fit(trainingData)

    println(model.coefficients)
    println(model.intercept)
    println(model.summary.r2)

    val predictions =  model.transform(testData)
    predictions.show()

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("r2")
    println(evaluator.evaluate(predictions))
  }
}
