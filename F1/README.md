```markdown
f1-race-predictor/
│
├── config/
│   ├── config.json
│   ├── features.json
│   └── model_params.json
│
├── data/
│   ├── raw/
│   │   └── fastf1_cache/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── logs/
│   ├── pipeline/
│   ├── training/
│   ├── prediction/
│   └── errors/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fastf1_client.py
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   └── data_splitter.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   ├── lap_features.py
│   │   ├── driver_features.py
│   │   ├── constructor_features.py
│   │   └── weather_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── baseline.py
│   │   ├── xgboost_model.py
│   │   ├── neural_net.py
│   │   └── model_factory.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── cross_validation.py
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_config.py
│   │   ├── metrics.py
│   │   └── plotting.py
│
├── experiments/
│   ├── experiments.ipynb
│   └── mlruns/
│
├── reports/
│   ├── latex/
│   │   ├── main.tex
│   │   ├── sections/
│   │   │   ├── introduction.tex
│   │   │   ├── data.tex
│   │   │   ├── methodology.tex
│   │   │   ├── results.tex
│   │   │   └── conclusions.tex
│   │   └── figures/
│   │
│   └── generated/
│       └── predictions_gp_YYYY.pdf
│
├── scripts/
│   ├── run_pipeline.py
│   ├── train_model.py
│   └── predict_race.py
│
├── requirements.txt
├── README.md
└── .gitignore
```