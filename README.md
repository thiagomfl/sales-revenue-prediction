# 📈 Sales Revenue Prediction

A machine learning project that predicts sales revenue based on seller experience, number of sales, and seasonal factors. Built with **Clean Architecture** principles, featuring a **FastAPI** backend and **Streamlit** frontend.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)

## 🎯 Project Overview

This project was developed as a hands-on exercise in building end-to-end ML applications with production-ready architecture. It demonstrates:

- **Exploratory Data Analysis** with pandas and seaborn
- **Model comparison** between Linear and Polynomial Regression
- **Clean Architecture** for maintainable and testable code
- **REST API** for model serving
- **Interactive UI** for predictions

## 🏗️ Architecture

The project follows **Clean Architecture** principles, ensuring separation of concerns and independence from external frameworks.
```
┌─────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                        │
│  ┌─────────────┐    ┌─────────────────────────────────┐ │
│  │   FastAPI   │    │       ModelRepository           │ │
│  │   Routes    │    │   (loads model from disk)       │ │
│  └──────┬──────┘    └───────────────┬─────────────────┘ │
│         │                           │                    │
│         ▼                           ▼                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │              APPLICATION (Use Cases)              │   │
│  │            PredictRevenueUseCase                  │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │                    DOMAIN                         │   │
│  │    SalesPredictionInput, SalesPredictionOutput   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure
```
sales-prediction/
├── data/                   # Dataset
├── notebooks/              # EDA and model training
├── core/                   # Shared ML code (preprocessing, evaluation)
├── api/                    # FastAPI service
│   ├── domain/             # Entities (pure business rules)
│   ├── application/        # Use cases and DTOs
│   └── infrastructure/     # External interfaces (API, ML repository)
├── app/                    # Streamlit frontend
├── saved_models/           # Trained models (.joblib)
└── tests/                  # Unit and integration tests
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sales-prediction.git
cd sales-prediction

# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell
```

### Running the Notebook
```bash
poetry run poe notebook
```

### Running the API
```bash
poetry run poe api
```

The API will be available at `http://localhost:8000`. Access the interactive docs at `http://localhost:8000/docs`.

### Running the Streamlit App
```bash
poetry run poe app
```

The app will be available at `http://localhost:8501`.

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| `tempo_de_experiencia` | Seller's experience in months |
| `numero_de_vendas` | Number of sales in a specific period |
| `fator_sazonal` | Seasonality factor (1-10, where 10 = peak season) |
| `receita_em_reais` | Total revenue generated (target variable) |

## 🧪 Model Comparison

| Model | Features | MSE | R² |
|-------|----------|-----|-----|
| Linear Regression | experience, sales | TBD | TBD |
| Polynomial Regression | experience, sales, seasonal | TBD | TBD |

> Results will be updated after training.

## 🛠️ Development

### Available Commands
```bash
# Format code
poetry run poe format

# Run linter
poetry run poe lint

# Type checking
poetry run poe typecheck

# Run all checks (format + lint + typecheck)
poetry run poe check

# Run tests
poetry run poe test
poetry run poe test-unit
poetry run poe test-integration
```

### Code Quality Tools

- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **mypy** - Static type checking
- **unittest** - Testing framework

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict revenue from input features |
| `GET` | `/model/info` | Get model metadata |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "experience_months": 36,
    "number_of_sales": 50,
    "seasonal_factor": 7
  }'
```

### Example Response
```json
{
  "predicted_revenue": 5432.10,
  "model_used": "polynomial_regression",
  "confidence_score": 0.85
}
```

## 🤝 Contributing

Contributions are welcome! Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages.
```
feat(scope): add new feature
fix(scope): fix bug
docs: update documentation
refactor(scope): refactor code
test(scope): add tests
chore: maintenance tasks
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Thiago** - Senior Software Developer

- Experience with Python, TypeScript, Angular, and cloud technologies
- Currently working on globalization and internationalization systems
- Passionate about machine learning and data science

---

⭐ If you found this project helpful, please consider giving it a star!
