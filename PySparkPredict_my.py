import argparse
import os

from pyspark.sql import SparkSession

import mlflow


def process(spark, data_path, result):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param result: путь сохранения результата
    """
    data_df = spark.read.parquet(data_path)
    model = mlflow.spark.load_model(f'models:/k-veryutina/2')
    predict = model.transform(data_df)
    predict.write.parquet(result)


def main(data, result):
    spark = _spark_session()
    process(spark, data, result)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkPredict').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.parquet', help='Please set datasets path.')
    parser.add_argument('--result', type=str, default='result', help='Please set result path.')
    args = parser.parse_args()
    data = args.data
    result = args.result
    
    mlflow.set_tracking_uri("https://mlflow.lab.karpov.courses")
    
    main(data, result)
