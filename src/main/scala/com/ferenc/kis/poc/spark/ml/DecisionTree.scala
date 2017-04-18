package com.ferenc.kis.poc.spark.ml

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{array, udf}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object DecisionTree {

  val INPUT_PATH = "src/main/resources/data/iris.csv"
  val NULL_VALUE = "?"

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[4]").appName("decision-tree").getOrCreate()

    val schema = StructType(
      Seq(
        StructField("SEPAL_LENGTH", DoubleType, false),
        StructField("SEPAL_WIDTH", DoubleType, false),
        StructField("PETAL_LENGTH", DoubleType, false),
        StructField("PETAL_WIDTH", DoubleType, false),
        StructField("SPECIES", StringType, false)
      )
    )

    val df = spark.read
      .option("header", true)
      .option("nullValue", NULL_VALUE)
      .schema(schema)
      .csv(INPUT_PATH)
    df.show()

    val stringIndexer = new StringIndexer().setInputCol("SPECIES").setOutputCol("INDEXED")
    val siModel = stringIndexer.fit(df)
    val indexedDf = siModel.transform(df)

    indexedDf.show()
    indexedDf.groupBy("SPECIES", "INDEXED").count().show()

    val columnNames = schema.filter(_.dataType != StringType).map(_.name)

    // You can plot correlation between variables in Zeppelin with SparkSQL
    columnNames.foreach(c =>
      println(s"Correlation between INDEXED and $c is ${indexedDf.stat.corr("INDEXED", c)}")
    )

    import spark.implicits._

    val denseVector = udf( (features: Seq[Double]) => Vectors.dense(features.toArray) )
    val mlData = indexedDf.select($"INDEXED" as "label", denseVector(array($"SEPAL_LENGTH", $"SEPAL_WIDTH", $"PETAL_LENGTH", $"PETAL_WIDTH")) as "features")

    val Array(trainingData, testData) = mlData.randomSplit(Array(0.7, 0.3))

    val decisionTreeClassifier = new DecisionTreeClassifier().setMaxDepth(6)
    val model = decisionTreeClassifier.fit(trainingData)

    println(model.toDebugString)

    println(model.numNodes)
    println(model.depth)
    println(model.numClasses)
    println(model.numFeatures)

    val rawPredictions = model.transform(testData)

    val labelConverter = new IndexToString().setInputCol("label").setOutputCol("labelString").setLabels(siModel.labels)
    val predictionConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictionString").setLabels(siModel.labels)

    val predictions = predictionConverter.transform(labelConverter.transform(rawPredictions))

    rawPredictions.show()
    predictions.show()

    val evaluator = new MulticlassClassificationEvaluator().setPredictionCol("prediction").setLabelCol("label").setMetricName("accuracy")
    println(s"Accuracy=${evaluator.evaluate(predictions)}")

    println("Confusion matrix")
    predictions.groupBy("labelString", "predictionString").count().show()
  }
}
