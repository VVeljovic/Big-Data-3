@echo off
docker exec spark-master sh -c "apk add --no-cache py3-numpy"
docker exec -i namenode bash -c "hdfs dfsadmin -safemode leave"
docker exec namenode hdfs dfs -rm -r /output
docker exec namenode hdfs dfs -mkdir -p /output
docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 /opt/spark-apps/consumer.docker.py "tumbling" "1 hour" "1 hour"
pause