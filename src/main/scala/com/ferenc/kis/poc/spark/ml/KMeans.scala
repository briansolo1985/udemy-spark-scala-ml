package com.ferenc.kis.poc.spark.ml

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors.dense
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType}

object KMeans {
  val INPUT_PATH = "src/main/resources/data/auto-data.csv"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName("naive-bayes").getOrCreate()

    val df = spark.read
      .option("header", true)
      .csv(INPUT_PATH)

    df.show()

    import df.sparkSession.implicits._
    val autoVectors = df.select(
      when($"doors" === "two", 0.0).otherwise(1.0).cast(DoubleType).as("doors"),
      when($"body" === "sedan", 0.0).otherwise(1.0).cast(DoubleType).as("body"),
      $"hp".cast(DoubleType),
      $"rpm".cast(DoubleType),
      $"mpg-city".cast(DoubleType).as("mpg")
    )

    autoVectors.show()
    println(autoVectors.count())

    val meanVal = autoVectors.agg(avg($"doors"), avg($"body"), avg($"hp"), avg($"rpm"), avg($"mpg"))
    val stdVal = autoVectors.agg(stddev($"doors"), stddev($"body"), stddev($"hp"), stddev($"rpm"), stddev($"mpg"))

    meanVal.show()
    stdVal.show()

    val denseVector = udf( (features: Seq[Double]) => dense(features.toArray) )

    val mlReady = autoVectors.join(meanVal)
      .join(stdVal)
      .select(
        lit(1.0).as("label"),
        denseVector(
          array(
            ($"doors" - $"avg(doors)") / $"stddev_samp(doors)",
            ($"body" - $"avg(body)") / $"stddev_samp(body)",
            ($"hp" - $"avg(hp)") / $"stddev_samp(hp)",
            ($"rpm" - $"avg(rpm)") / $"stddev_samp(rpm)",
            ($"mpg" - $"avg(mpg)") / $"stddev_samp(mpg)"
          )
        ).as("features")
      )

    mlReady.show()

    val kMeans = new KMeans().setK(4).setSeed(1L)

    // Perform K-Means Clustering
    val model = kMeans.fit(mlReady)
    val predictions = model.transform(mlReady)

    predictions.printSchema()

    val denseVectorToArray = udf( (features: Vector) => features.toArray )

    predictions.select(denseVectorToArray($"features").as("features"), $"prediction")
      .select($"features".getItem(0).as("n_doors"), $"features".getItem(1).as("n_body"),
        $"features".getItem(2).as("n_hp"), $"features".getItem(3).as("n_rpm"),
        $"features".getItem(4).as("n_mpg"), $"prediction")
      .join(meanVal)
      .join(stdVal)
      .select(
        ($"n_doors" * $"stddev_samp(doors)" + $"avg(doors)").as("doors"),
        ($"n_body" * $"stddev_samp(body)" + $"avg(body)").as("body"),
        ($"n_hp" * $"stddev_samp(hp)" + $"avg(hp)").as("hp"),
        ($"n_rpm" * $"stddev_samp(rpm)" + $"avg(rpm)").as("rpm"),
        ($"n_mpg" * $"stddev_samp(mpg)" + $"avg(mpg)").as("mpg"),
        $"prediction"
      )
      .show()

    predictions.groupBy("prediction").count().show()
  }
}