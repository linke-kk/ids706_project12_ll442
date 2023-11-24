from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col
import pyspark.sql.functions as F
from collections import defaultdict


## step1: ingest data from dbfs with spark
# Start a Spark session
spark = SparkSession.builder.appName("CSVIngestion").getOrCreate()

# Read CSV file
df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(
        "dbfs:/FileStore/shared_uploads/linkekk@ad.unc.edu/OccupationalEmploymentandWageStatistics_final.csv"
    )
)

# Show the first few rows of the DataFrame
df.head()


## step2: prepare the data: convert csv file to a table
df.createOrReplaceTempView("kk_table")

result = spark.sql("SELECT * FROM kk_table WHERE Occupation = 'Legal'")
result.show()

## step3: Analyze the data find the mean, max, min value

myHash = defaultdict(list)
meanHash = defaultdict()
maxHash = defaultdict()
minHash = defaultdict()

columns = [
    "10th %",
    "25th %",
    "Entry level",
    "Median",
    "Mean",
    "Experienced",
    "75th %",
    "90th %",
]

for column in columns:
    df = df.withColumn(column, regexp_replace(col(column), "[\$,]", "").cast("integer"))
    # meanHash[column] = int(statistics.mean(myHash[column]))
    # maxHash[column] = int(max(myHash[column]))
    # minHash[column] = int(min(myHash[column]))
for column in columns:
    meanHash[column] = df.select(F.mean(col(column))).first()[0]
    maxHash[column] = df.select(F.max(col(column))).first()[0]
    minHash[column] = df.select(F.min(col(column))).first()[0]
# print(myHash)
print("Mean Value information")
print(meanHash)
print("Max Value information")
print(maxHash)
print("Min Value information")
print(minHash)
