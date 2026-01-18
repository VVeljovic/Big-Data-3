from confluent_kafka import Producer
from uuid import uuid4
import time

config = {
    'bootstrap.servers': 'localhost:9092'
}

producer = Producer(config)

file = open('us_accidents_cleaned.csv', 'r', encoding='utf-8')

line = file.readline()
for line in file:
    producer.produce(
        topic = 'us_accidents_topic',
        key = str(uuid4()),
        value = line.strip()
    )
    time.sleep(1)
    print(line)
    producer.flush()

file.close()
