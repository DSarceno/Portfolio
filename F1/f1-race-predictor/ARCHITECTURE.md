# Architecture Documentation

## Design Decisions

### 1. Temporal Data Split
- NOT random split (would leak future information)
- Train: 2020-2023, Validation: 2024, Test: 2025
- Respects realistic prediction scenario

### 2. Feature Engineering Strategy
- Generate 50+ features initially
- Feature selection reduces to 20-30 best
- Features split by category: lap, driver, constructor, weather
- Weather ALWAYS included (critical for F1)

### 3. Multiple Models
- **XGBoost**: Best with tabular data, interpretable
- **Neural Network**: Potential for complex patterns
- **Baseline**: Lower bound, simple grid position predictor
- Comparison: Validates results, shows model capability

### 4. Logging Strategy
- Logging module (not MLflow, not print)
- Centralized logging config
- Per-module loggers
- File + console output

### 5. Data Pipeline
- Extract → Validate → Transform → Feature Engineering → Split
- Each step documented and testable
- Caching to avoid re-downloading

### 6. API Design
- REST endpoints (not WebSocket)
- Request validation with Pydantic
- Stateless service (models loaded at startup)
- Docker-ready

## Class Hierarchy

```
BaseModel (ABC)
├── XGBoostModel
├── NeuralNetModel
└── BaselineModel

DataLoader
├── FastF1Client
├── DataValidator
└── DataSplitter

FeatureBuilder
├── LapFeaturesExtractor
├── DriverFeaturesExtractor
├── ConstructorFeaturesExtractor
└── WeatherFeaturesExtractor

Trainer
├── Evaluator
└── TemporalCrossValidator
```

## Data Flow

```
FastF1 API
    ↓
data/raw/ (cached)
    ↓
FastF1Client.get_session()
    ↓
DataLoader.load_race()
    ↓
DataValidator.validate()
    ↓
FeatureBuilder.build_features()
    ↓
DataSplitter.temporal_split()
    ↓
data/processed/
    ↓
Trainer.train_single_model()
    ↓
models/best_model.pkl
    ↓
Predictor.predict_race()
    ↓
outputs/predictions.json
```

## Key Patterns

1. **Factory Pattern**: `ModelFactory` for creating model instances
2. **Strategy Pattern**: Different feature extractors per category
3. **Pipeline Pattern**: Sequential data processing steps
4. **Singleton**: `Config` loaded once per process
5. **Template Method**: `BaseModel` defines the training/prediction contract

## Error Handling

- Specific exception types caught at each layer
- All errors logged with `exc_info=True` for stack traces
- Graceful degradation: missing telemetry returns NaN features
- User-friendly error messages in API responses

## Performance Considerations

- Caching FastF1 data locally (avoids re-downloading)
- Vectorized operations via numpy/pandas
- Batch processing for model inference
- Early stopping for neural networks and XGBoost

## Testing Strategy

- Unit tests for each module with pytest
- Fixtures in `tests/conftest.py` for shared test data
- Mock FastF1Client to avoid network calls in tests
- Coverage target: 80%+

---

**For questions, see [src/models/base_model.py](src/models/base_model.py) for pattern examples.**
