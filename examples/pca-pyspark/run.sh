#!/usr/bin/env bash

source ../../conf/env.sh

# CSV data is the same as in Spark example "ml/pca_example.py"
# The data file should be copied to $HDFS_ROOT before running examples
DATA_FILE=data/pca_data.csv

APP_PY=pca-pyspark.py
K=3

time $SPARK_HOME/bin/spark-submit --master $SPARK_MASTER -v \
    --num-executors $SPARK_NUM_EXECUTORS \
    --driver-memory $SPARK_DRIVER_MEMORY \
    --executor-cores $SPARK_EXECUTOR_CORES \
    --executor-memory $SPARK_EXECUTOR_MEMORY \
    --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.default.parallelism=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.sql.shuffle.partitions=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.driver.extraClassPath=$SPARK_DRIVER_CLASSPATH" \
    --conf "spark.executor.extraClassPath=$SPARK_EXECUTOR_CLASSPATH" \
    --conf "spark.shuffle.reduceLocality.enabled=false" \
    --conf "spark.network.timeout=1200s" \
    --conf "spark.task.maxFailures=1" \
    --jars $OAP_MLLIB_JAR \
    $APP_PY $DATA_FILE $K \
    2>&1 | tee PCA-$(date +%m%d_%H_%M_%S).log
