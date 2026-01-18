from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder \
    .appName("AccidentDataTrainer") \
    .getOrCreate()

df = spark.read.csv(
    "us_accidents_cleaned.csv",
    header=True,
    inferSchema=True
)

df = df.withColumn(
    "label",
    when(col("Severity") >= 3, 1).otherwise(0)
)

fractions = {0: 0.06, 1: 1.0}
df = df.sampleBy("label", fractions, seed=36)

train_df, test_df = df.randomSplit([0.85, 0.15], seed=5043)

categorical_cols = [
    "Weather_Condition",
    "Wind_Direction",
    "Sunrise_Sunset"
]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "Indexed",
        handleInvalid="keep"
    )
    for c in categorical_cols
]


feature_cols = [
    "Visibility_mi", "Wind_Speed_mph",
    "Temperature_F", "Humidity_percent", "Pressure_in",
    "Distance_mi", "Precipitation_in",
    "Start_Lat", "Start_Lng",

    "Junction", "Traffic_Signal", "Crossing",

    "Weather_ConditionIndexed",
    "Wind_DirectionIndexed",
    "Sunrise_SunsetIndexed"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=30,
    maxDepth=5,
    maxBins=150,
    seed=36
)

pipeline = Pipeline(
    stages=indexers + [assembler, gbt]
)

model = pipeline.fit(train_df)

predictions = model.transform(test_df)

predictions.select(
    "label", "prediction", "probability"
).show(5, truncate=False)

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print(f"Procena modela: {auc}")

model.write().overwrite().save("models/accident_gbt_pipeline")
