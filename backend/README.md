# Microgrid Backend API

Python Flask-based REST API for microgrid stability prediction and analysis.

## Structure

```
backend/
├── src/
│   ├── api/          # REST API routes and schemas
│   ├── data/         # Data parsing, validation, and preprocessing
│   ├── models/       # Prediction models (Classical, LSTM, Advanced)
│   ├── simulation/   # Microgrid simulator and EMS controller
│   ├── analysis/     # Comparative analysis and results export
│   ├── config/       # Configuration management
│   └── app.py        # Flask application entry point
├── tests/
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── property/     # Property-based tests with Hypothesis
├── requirements.txt  # Python dependencies
└── setup.py          # Package setup
```

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install package in development mode:
   ```bash
   pip install -e .
   ```

## Running

Start the development server:
```bash
python src/app.py
```

Server will run on `http://localhost:5000`

## Testing

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run specific test types:
```bash
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/property/      # Property-based tests only
```

## Code Quality

Format code:
```bash
black src/ tests/
```

Lint code:
```bash
ruff check src/ tests/
```

Type check:
```bash
mypy src/
```

## API Endpoints

See PROJECT_STRUCTURE.md in the root directory for complete API documentation.

## Development

Modules will be implemented in the following order:
1. Data layer (parser, validator, pipeline)
2. Model layer (base predictor, classical, LSTM, advanced)
3. Simulation layer (simulator, EMS controller, stability analyzer)
4. Analysis layer (comparative engine, results exporter)
5. API layer (routes, schemas, error handling)
