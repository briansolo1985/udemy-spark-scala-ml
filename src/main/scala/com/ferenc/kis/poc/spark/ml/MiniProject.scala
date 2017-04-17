package com.ferenc.kis.poc.spark.ml

import org.apache.spark.sql.SparkSession
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
      .withColumn("ROUNDED_AGE", bround($"AGE", -1))
    custidWithNumericFeatures.show()

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
    features.show(10000)
    features.printSchema()
    features.columns.foreach(c1 => {
      features.columns.foreach(c2 => {
        if(c1 != c2) {
          println(s"Correlation between $c1 and $c2 is ${features.stat.corr(c1, c2)}")
        }
      })
    })

  }
}
