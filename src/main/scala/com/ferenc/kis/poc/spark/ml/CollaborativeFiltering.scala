package com.ferenc.kis.poc.spark.ml

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType}

object CollaborativeFiltering {

  val INPUT_PATH = "d:\\Projects\\SparkML\\data\\UserItemData.txt"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName("collaborative-filtering").getOrCreate()

    val df = spark.read
      .option("header", false)
      .csv(INPUT_PATH)

    df.show()

    import spark.implicits._
    val ratingsDf = df.select(
      $"_c0".cast(IntegerType).as("user"),
      $"_c1".cast(IntegerType).as("item"),
      $"_c2".cast(DoubleType).as("rating")
    )

    ratingsDf.show()

    val Array(trainingData, testData) = ratingsDf.randomSplit(Array(0.9, 0.1))

    val als = new ALS().setRank(10).setMaxIter(5).setUserCol("user").setItemCol("item").setRatingCol("rating")

    val model = als.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.show()
  }
}
