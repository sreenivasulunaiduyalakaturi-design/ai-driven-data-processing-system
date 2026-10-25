# AI-Driven Data Processing System

Pipeline: Load -> Clean -> Transform -> AI Analysis (anomalies, clusters, optional prediction) -> Report

## Setup
    pip install -r requirements.txt

## Run
    python main.py                                    # demo with sample data
    python main.py --input mydata.csv                 # your own CSV
    python main.py --input mydata.csv --target price  # also trains a model

Results are saved in `output/` (processed_data.csv, summary.json).
