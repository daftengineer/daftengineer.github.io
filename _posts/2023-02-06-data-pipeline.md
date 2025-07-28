---
title: A Data Ingestion Pipeline to Take in Data from API Source of Simulated Wind Farm to TimescaleDB
tags: rabitmq timescaledb go python fastapi pydantic
---

I built this wind farm data pipeline as a proof of concept for ingesting sensor data at scale. The setup uses FastAPI, RabbitMQ, and TimescaleDB - all containerized so you can spin it up with one command.

Full code is at [github.com/daftengineer/data-ingestion](https://github.com/daftengineer/data-ingestion)

Once it is built you can just do 
```
docker-compose up --build
```

## How It Works

The sample dataset only had 3 hours of data, so I couldn't test the time-series aggregation features I wanted to try. Also, this was my first time with TimescaleDB, so I'm definitely not using all its capabilities (those compression functions look interesting though).

Data flow: API request → FastAPI server → RabbitMQ → consumer → TimescaleDB

To ingest data from a CSV file:
```
curl -F "file=@solarfarm01.csv;type=text/csv" http://127.0.0.1:8000/ingest_from_file -X POST -H "customer_name: adani" -H "asset_name: A01"
```

2. To retrieve datapoint
```
curl -X POST http://127.0.0.1:8000/retrive_datapoints -d '{"asset":"C01","timeseries":["generator___mechanical___speed","inverter___temperature"],"daterange":["2012-06-30","2012-07-03"]}'
```

3. To list the customers
```
curl -X GET http://127.0.0.1:8000/list_customers
```

4. Edit customer description
```
curl -X POST http://127.0.0.1:8000/edit_customer_information -d '{"customer_name":"reliance","updated_description":"test"}'
```

5. To Compute stats
```
curl -X POST http://127.0.0.1:8000/compute_stats -d '{"customer_name":"reliance","timeseries":"generator___mechanical___speed","daterange":["2012-06-30","2012-07-03"]}'
```

6. verify time series date in date range
```
curl -X POST http://127.0.0.1:8000/verify_date_ts -d '{"timeseries":"generator___mechanical___speed","daterange":["2012-06-30","2012-07-03"]}'
```

7. Delete data
```
curl -X POST http://127.0.0.1:8000/delete_datapoints -d '{"timeseries":"generator___mechanical___speed", "daterange":["2012-07-01 09:00:01","2012-07-01 10:06:45"]}'
```
