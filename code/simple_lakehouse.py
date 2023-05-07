from datetime import datetime
from typing import Optional
import findspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import to_date, lit, input_file_name, split, regexp_replace, sha2, concat_ws, rank, desc, \
    year, month
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType


def get_spark() -> SparkSession:
    """
    Get a SparkSession
    :return:
    """
    findspark.init()

    sc = SparkContext.getOrCreate(
        SparkConf()
        .setMaster("local[1]")
        .set("spark.jars.packages", "io.delta:delta-core_2.12:2.3.0")
        .set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .setAppName('simple_lakehouse')
    )

    spark = SparkSession(sc)

    return spark


def get_schema() -> StructType:
    """
    Get the schema for the data
    :return:
    """
    return StructType([
        StructField('date', DateType()),
        StructField('name', StringType()),
        StructField('phone', StringType()),
        StructField('email', StringType()),
        StructField('country', StringType()),
        StructField('colour', StringType()),
        StructField('currency', StringType())
    ])


def read_raw_data(
        path: str,
        schema: StructType,
        spark: SparkSession
) -> DataFrame:
    """
    Read the raw data
    :param path: Path to the data
    :param schema: Schema for the data
    :param spark: SparkSession
    :return:
    """
    return (
        spark.read
        .format("csv")
        .options(
            header="true",
            inferSchema="true"
        )
        .load(path, schema=schema)
    )


def write_to_lakehouse(
        df: DataFrame,
        path: str,
        partition: Optional[bool] = False
) -> None:
    """
    Write a DataFrame to a Delta Lake
    :param df: DataFrame to write
    :param path: Path to write to
    :param partition: Partition the data
    :return:
    """
    save = (
        df.write
        .format("delta")
        .options(
            mergeSchema=True,
            overwriteSchema=True
        )
        .mode("overwrite")
    )

    if partition:
        save = save.partitionBy("batch_date")

    save.save(path)


def read_from_lakehouse(
        path: str,
        spark: SparkSession,
        batch_date: Optional[str] = None
) -> DataFrame:
    """
    Read a DataFrame from a Delta Lake
    :param path: Path to read from
    :param spark: SparkSession
    :param batch_date: Batch date to read
    :return:
    """
    df = (
        spark.read
        .format("delta")
        .load(path)
    )

    if batch_date:
        df = df.filter(df.batch_date == batch_date)

    return df


def clean_bronze_data(
        df: DataFrame
) -> DataFrame:
    """
    Clean the bronze data
    :param df: DataFrame to clean
    :return: DataFrame
    """
    split_name = split(df['name'], ' ')
    df = df.withColumn('first_name', split_name.getItem(0))
    df = df.withColumn('last_name', split_name.getItem(1))
    df = df.withColumn('currency', regexp_replace('currency', ' ', '').cast(IntegerType()))

    df = df.drop('name')

    return df


def extract_person_dimension(
        df: DataFrame
) -> DataFrame:
    """
    Extract the person dimension
    :param df: DataFrame to extract from
    :return: DataFrame
    """
    df = df.select(
        'first_name',
        'last_name',
        'phone',
        'email',
        'country'
    ).distinct()

    df = df.withColumn('person_id', sha2(
        concat_ws(
            '',
            df['first_name'],
            df['last_name'],
            df['phone'],
            df['email']
        ),
        256
    ))

    df = df.select(
        'person_id',
        'first_name',
        'last_name',
        'phone',
        'email',
        'country'
    )

    return df


def extract_country_dimension(
        df: DataFrame
) -> DataFrame:
    """
    Extract the country dimension
    :param df: DataFrame to extract from
    :return: DataFrame
    """
    df = df.select(
        'country'
    ).distinct()

    df = df.withColumn('country_id', sha2('country', 256))

    df = df.select(
        'country_id',
        'country'
    )

    return df


def extract_colour_dimension(
        df: DataFrame
) -> DataFrame:
    """
    Extract the colour dimension
    :param df: DataFrame to extract from
    :return: DataFrame
    """
    df = df.select(
        'colour'
    ).distinct()

    df = df.withColumn('colour_id', sha2('colour', 256))

    df = df.select(
        'colour_id',
        'colour'
    )

    return df


def create_fact_table(
        df: DataFrame,
        person_df: DataFrame,
        country_df: DataFrame,
        colour_df: DataFrame
) -> DataFrame:
    """
    Create the fact table
    :param df: DataFrame to create from
    :param person_df: Person dimension DataFrame
    :param country_df: Country dimension DataFrame
    :param colour_df: Colour dimension DataFrame
    :return:
    """

    df = df.join(
        person_df,
        [
            df.first_name == person_df.first_name,
            df.last_name == person_df.last_name,
            df.phone == person_df.phone,
            df.email == person_df.email,
            df.country == person_df.country
        ],
        how='inner'
    )

    df = df.join(
        country_df,
        'country',
        how='left'
    )

    df = df.join(
        colour_df,
        'colour',
        how='left'
    )

    df = df.select(
        'date',
        'person_id',
        'country_id',
        'colour_id',
        'currency'
    )

    return df


def main() -> None:
    spark = get_spark()

    schema = get_schema()

    ### Bronze Layer Begin ###
    raw_df = read_raw_data(
        path="data/sample_data.csv",
        schema=schema,
        spark=spark
    )

    # Add batch date and input file name to raw data
    batch_date = datetime.now().strftime("%Y-%m-%d")
    raw_df = raw_df.withColumn(
        "batch_date",
        to_date(
            lit(batch_date),
            "yyyy-MM-dd"
        )
    )
    raw_df = raw_df.withColumn(
        "input_file",
        input_file_name()
    )

    # Write raw data to lakehouse bronze layer
    write_to_lakehouse(
        df=raw_df,
        path="lakehouse/bronze/bronze_data",
        partition=True
    )
    ### Bronze Layer End ###

    ### Silver Layer Begin ###
    # Read raw data from lakehouse bronze layer
    bronze_df = read_from_lakehouse(
        path="lakehouse/bronze/bronze_data",
        spark=spark,
        batch_date=batch_date
    )

    bronze_df = clean_bronze_data(bronze_df)

    person_df = extract_person_dimension(bronze_df)
    country_df = extract_country_dimension(bronze_df)
    colour_df = extract_colour_dimension(bronze_df)

    fact_df = create_fact_table(
        bronze_df,
        person_df,
        country_df,
        colour_df
    )

    # Write dimensions to lakehouse silver layer
    write_to_lakehouse(
        df=person_df,
        path="lakehouse/silver/person"
    )

    write_to_lakehouse(
        df=country_df,
        path="lakehouse/silver/country"
    )

    write_to_lakehouse(
        df=colour_df,
        path="lakehouse/silver/colour"
    )

    # Write fact table to lakehouse gold layer
    write_to_lakehouse(
        df=fact_df,
        path="lakehouse/silver/fact_table"
    )
    ### Silver Layer End ###

    ### Gold Layer Begin ###
    person_df = read_from_lakehouse(
        path="lakehouse/silver/person",
        spark=spark
    )
    country_df = read_from_lakehouse(
        path="lakehouse/silver/country",
        spark=spark
    )
    colour_df = read_from_lakehouse(
        path="lakehouse/silver/colour",
        spark=spark
    )
    fact_df = read_from_lakehouse(
        path="lakehouse/silver/fact_table",
        spark=spark
    )

    spent_per_day_df = (
        fact_df
        .groupBy('date')
        .sum('currency')
        .withColumnRenamed(
            'sum(currency)',
            'spent_per_day'
        )
    )
    person_spent_df = fact_df.groupBy('person_id').sum('currency').withColumnRenamed('sum(currency)', 'person_spent')
    person_spent_df = person_spent_df.join(person_df, 'person_id', how='inner')
    person_spent_df = person_spent_df.select(
        'person_id',
        'first_name',
        'last_name',
        'phone',
        'email',
        'country',
        'person_spent'
    )
    most_bought_colour_per_month_df = (
        fact_df
        .withColumn('year', year('date'))
        .withColumn('month', month('date'))
        .groupBy('year', 'month', 'colour_id')
        .count()
        .withColumnRenamed('count', 'colour_count')
        .withColumn('rank', rank().over(
            Window.partitionBy(
                'year',
                'month'
            ).orderBy(desc('colour_count'))
        ))
    )
    most_bought_colour_per_month_df = most_bought_colour_per_month_df.join(
        colour_df,
        'colour_id',
        how='inner'
    ).select(
        'year',
        'month',
        'colour_id',
        'colour',
        'colour_count',
        'rank'
    )

    # Write aggregated data to lakehouse gold layer
    write_to_lakehouse(
        df=spent_per_day_df,
        path="lakehouse/gold/spent_per_day"
    )

    write_to_lakehouse(
        df=person_spent_df,
        path="lakehouse/gold/person_spent"
    )

    write_to_lakehouse(
        df=most_bought_colour_per_month_df,
        path="lakehouse/gold/most_bought_colour_per_month"
    )
    ### Gold Layer End ###


if __name__ == "__main__":
    main()
