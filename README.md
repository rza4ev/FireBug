# FireBug\
🔥 Wildfire AI – Real-Time Fire Risk Detection System

IoT Sensors → Apache Kafka Streaming → XGBoost ML Model → Real-Time Streamlit Dashboard

📌 Overview

This project is an end-to-end real-time wildfire risk detection system that processes sensor data using a streaming architecture.
The system ingests sensor readings (temperature, humidity, gas levels, wind, etc.) via Apache Kafka, sends them to an XGBoost machine learning model, and visualizes predictions on an interactive Streamlit Dashboard.

The pipeline supports live monitoring, early fire risk prediction, alert generation, and geospatial visualization.

🔧 System Architecture

1. Fire Sensors (Simulated or Real IoT Devices)
Sensors generate environmental data such as temperature, humidity, CO/CO₂, VOC, wind speed/direction, and solar radiation.

2. Apache Kafka — Data Streaming Layer

sensor.readings → raw sensor values

fire.risk.predictions → ML model outputs

fire.alerts → critical fire alerts (threshold-based)
Kafka ensures scalable, fault-tolerant, and real-time message delivery.

3. ML Prediction Service (XGBoost Model)
A Python microservice consumes Kafka events, processes sensor features, and predicts:

Probability of ignition in the next 5 minutes

Risk score (0–100%)

Alert flag (True/False)

Model logic:

prob_fire, alert = predict_risk(sensor_reading)


4. Streamlit Dashboard — Real-Time Visualization
The dashboard displays:

Live fire risk predictions

Alert notifications

Sensor time series

Geo-mapped sensor locations (PyDeck)

Analytics: correlation heatmaps, time-series charts, region-based filtering

OSM road-network overlays (optional via OSMnx)

🚀 Features

✔ Real-time data streaming with Kafka
✔ ML-powered fire risk prediction (XGBoost)
✔ Threshold-based alerting system
✔ Live updating dashboards
✔ Geospatial visualization with PyDeck + OSMnx
✔ Automatic dataset simulation for demo mode
✔ Modular, production-ready design

💡 Demo Workflow

Sensor data is streamed into Kafka.

ML Consumer reads messages and computes risk scores.

Predictions and alerts are published to Kafka topics.

Streamlit dashboard visualizes everything in real time.

Pipeline:

Sensors → Kafka → ML Model → Kafka → Streamlit Dashboard

📁 Project Structure
/ml_consumer.py        → Kafka consumer producing ML predictions
/predict_service.py    → XGBoost model loading & prediction logic
/dashboard.py          → Streamlit real-time dashboard
fire_ignition_xgb_model.joblib
fire_model_config.json
dataset.csv

📦 Technologies Used

Python

Apache Kafka (Message Streaming)

XGBoost (ML Model)

Streamlit (Dashboard UI)

PyDeck (Geospatial visualization)

Pandas / NumPy / Joblib

OSMNx (optional) for road-network visualization

🧪 Running the System
# Start Kafka
docker-compose up

# Run ML consumer
python ml_consumer.py

# Start dashboard
streamlit run dashboard.py

🎯 Goal

The goal of this system is to deliver an AI-powered wildfire monitoring platform that supports:

Early fire detection

Real-time anomaly tracking

Fast emergency response

Scalable IoT data ingestion
