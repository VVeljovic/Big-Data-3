from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, TimestampType
from pyspark.sql.functions import from_csv, avg, window, col, count, collect_list, udf
from pyspark.ml import PipelineModel
from collections import Counter
import sys

window_type = sys.argv[1]
window_duration = sys.argv[2]
slide_duration = sys.argv[3]

spark = SparkSession.builder.appName("Big-Data-3") \
        .getOrCreate()

model = PipelineModel.load("hdfs://namenode:9000/models/accident_gbt_pipeline")

accidents_schema = StructType([
    StructField("ID", StringType(), True),
    StructField("Source", StringType(), True),
    StructField("Severity", IntegerType(), True),
    StructField("Start_Time", TimestampType(), True),
    StructField("End_Time", TimestampType(), True),
    StructField("Start_Lat", DoubleType(), True),
    StructField("Start_Lng", DoubleType(), True),
    StructField("End_Lat", DoubleType(), True),
    StructField("End_Lng", DoubleType(), True),
    StructField("Distance_mi", DoubleType(), True),
    StructField("Description", StringType(), True),
    StructField("Street", StringType(), True),
    StructField("City", StringType(), True),
    StructField("County", StringType(), True),
    StructField("State", StringType(), True),
    StructField("Zipcode", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("Timezone", StringType(), True),
    StructField("Airport_Code", StringType(), True),
    StructField("Weather_Timestamp", TimestampType(), True),
    StructField("Temperature_F", DoubleType(), True),
    StructField("Wind_Chill_F", DoubleType(), True),
    StructField("Humidity_percent", DoubleType(), True),
    StructField("Pressure_in", DoubleType(), True),
    StructField("Visibility_mi", DoubleType(), True),
    StructField("Wind_Direction", StringType(), True),
    StructField("Wind_Speed_mph", DoubleType(), True),
    StructField("Precipitation_in", DoubleType(), True),
    StructField("Weather_Condition", StringType(), True),
    StructField("Amenity", BooleanType(), True),
    StructField("Bump", BooleanType(), True),
    StructField("Crossing", BooleanType(), True),
    StructField("Give_Way", BooleanType(), True),
    StructField("Junction", BooleanType(), True),
    StructField("No_Exit", BooleanType(), True),
    StructField("Railway", BooleanType(), True),
    StructField("Roundabout", BooleanType(), True),
    StructField("Station", BooleanType(), True),
    StructField("Stop", BooleanType(), True),
    StructField("Traffic_Calming", BooleanType(), True),
    StructField("Traffic_Signal", BooleanType(), True),
    StructField("Turning_Loop", BooleanType(), True),
    StructField("Sunrise_Sunset", StringType(), True),
    StructField("Civil_Twilight", StringType(), True),
    StructField("Nautical_Twilight", StringType(), True),
    StructField("Astronomical_Twilight", StringType(), True)
])

@udf(StringType())
def mode_udf(values):
    if not values:
        return None
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]

string_schema = accidents_schema.simpleString()

df = (
    spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "us_accidents_topic") \
    .option("startingOffsets", "earliest") \
    .load() \
    .selectExpr("CAST(value AS STRING) as csv_line") \
)

parsed_df = df.select(from_csv(df.csv_line, string_schema).alias("data")).select("data.*")

if (window_type.lower() == "tumbling"):
    windowed_col = window(col("Start_Time"), window_duration)
else:
    windowed_col = window(col("Start_Time"), window_duration, slide_duration)


windowed_df = (
    parsed_df.withWatermark("Start_Time", "1 hour")
    .groupBy(windowed_col)
    .agg(
        avg("Visibility_mi").alias("Visibility_mi"),
        avg("Wind_Speed_mph").alias("Wind_Speed_mph"),
        avg("Temperature_F").alias("Temperature_F"),
        avg("Humidity_percent").alias("Humidity_percent"),
        avg("Pressure_in").alias("Pressure_in"),
        avg("Distance_mi").alias("Distance_mi"),
        avg("Precipitation_in").alias("Precipitation_in"),
        avg("Start_Lat").alias("Start_Lat"),
        avg("Start_Lng").alias("Start_Lng"),
        avg(col("Junction").cast("double")).alias("Junction"),
        avg(col("Traffic_Signal").cast("double")).alias("Traffic_Signal"),
        avg(col("Crossing").cast("double")).alias("Crossing"),
        mode_udf(collect_list("Weather_Condition")).alias("Weather_Condition"),
        mode_udf(collect_list("Wind_Direction")).alias("Wind_Direction"),
        mode_udf(collect_list("Sunrise_Sunset")).alias("Sunrise_Sunset"),
        count("*").alias("accident_count"),
    )
)

windowed_features = windowed_df.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    "accident_count",
    "Visibility_mi",
    "Wind_Speed_mph",
    "Temperature_F",
    "Humidity_percent",
    "Pressure_in",
    "Distance_mi",
    "Precipitation_in",
    "Start_Lat",
    "Start_Lng",
    "Junction",
    "Traffic_Signal",
    "Crossing",
    "Weather_Condition",
    "Wind_Direction",
    "Sunrise_Sunset"
)

predictions = model.transform(windowed_features)

predictions.writeStream \
 .format("json") \
 .outputMode("append") \
 .option("path", "hdfs://namenode:9000/output/predictions_json") \
 .option("checkpointLocation", "hdfs://namenode:9000/output/checkpoint") \
 .start() \
 .awaitTermination()