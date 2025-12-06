# ml_consumer.py

from kafka import KafkaConsumer, KafkaProducer
import json
from predict_service import predict_risk  # əvvəlki mesajda yazdığımız modul

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_IN = "sensor.readings"
TOPIC_PRED = "fire.risk.predictions"
TOPIC_ALERTS = "fire.alerts"

def main():
    consumer = KafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="ml-prediction-service",
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Listening to '{TOPIC_IN}' and writing to '{TOPIC_PRED}' / '{TOPIC_ALERTS}' ...")

    for msg in consumer:
        reading = msg.value  # bu, CSV-dən gələn 1 sətrin dict formasıdır

        # ML modeldən ehtimal və alert flag al
        prob_fire, alert = predict_risk(reading)

        prediction_event = {
            **reading,
            "p_ignite_next_5min": prob_fire,
            "p_ignite_percent": prob_fire * 100,
            "alert": alert,
        }

        # 1) prediction stream-ə yaz
        producer.send(TOPIC_PRED, value=prediction_event)

        # 2) əgər kritikdirsə, ayrıca alert topic-ə
        if alert:
            alert_event = {
                "datetime": reading.get("datetime"),
                "region": reading.get("region"),
                "lat": reading.get("lat"),
                "lon": reading.get("lon"),
                "p_ignite_next_5min": prob_fire,
                "p_ignite_percent": prob_fire * 100,
            }
            producer.send(TOPIC_ALERTS, value=alert_event)

        producer.flush()

        print(
            f"[ML] {reading.get('datetime')} / {reading.get('region')} "
            f"-> P(fire)={prob_fire*100:.2f}%, alert={alert}"
        )

if __name__ == "__main__":
    main()
