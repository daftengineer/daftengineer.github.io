---
title: A data ingestion pipeline to take in data from API source of simulated wind farm to timescaleDB
tags: TeXt
---

Here in this article I am showing the sample data pipeline that takes in the data from API in file or record format and send it to fastapi server to process the request and connect to timescaleDB. The message broker used here is rabbitMQ. Note: Pydantic Data model is pending.
All the code is packaged in dockerfiles and runs with just one command. Repo for the code can be found at [here](https://github.com/daftengineer/data-ingestion)

Once it is built you can just do 
```
docker-compose up --build
```

## Observations

The data I received had only 3 hours of data so multiday query and aggregates were difficult. I havent worked on timescaledb so I couldn't fully utilize it. (It has fancy sublinear functions that I am quite interested)
1. Flow of data. API req -> API_SERVER -> Rabbitmq -> consumer -> timescaledb. Below command should ingest the data for given dataset
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
