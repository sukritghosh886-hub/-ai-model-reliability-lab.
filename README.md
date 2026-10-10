# -ai-model-reliability-lab.


# AI Model Reliability Lab

A lightweight machine-learning engineering project for evaluating
binary classification predictions, investigating errors, and
reviewing prediction confidence.

## Features

- Accuracy, precision, recall, and F1-score
- Confusion matrix and error indices
- High-confidence incorrect prediction detection
- Input validation for labels and confidence values
- Automated reliability recommendations
- Built-in scikit-learn benchmark
- FastAPI endpoints and interactive Swagger documentation

## Architecture

Google Colab notebook -> GitHub -> FastAPI deployment

The application does not require a GPU. Its demonstration uses a
small dataset bundled with scikit-learn.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Project status |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation |
| GET | `/demo` | Run the built-in ML benchmark |
| POST | `/evaluate` | Evaluate supplied predictions |

## Installation

Python 3.11 is the recommended starting environment.

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open `http://127.0.0.1:7860/docs` to interact with the API locally.

For a cloud platform that supplies a PORT environment variable,
use its required start command, for example:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Example Request

Send a POST request to `/evaluate`:

```json
{
  "true_labels": [0, 1, 1, 0],
  "predicted_labels": [0, 1, 0, 0],
  "confidences": [0.91, 0.95, 0.88, 0.79]
}
```

The API returns classification metrics, a confusion matrix,
error indices, and confidence-related warnings.

## Testing

Use `/docs` to test the endpoints.

Check that:
- `/health` returns `{"status": "ok"}`.
- `/demo` returns benchmark metrics.
- `/evaluate` accepts valid binary labels.
- Invalid labels or mismatched list lengths are rejected.

## Limitations

This is an educational engineering demonstration, not a
production-certified reliability or safety system.

The confidence endpoint reviews caller-supplied scores. It does not
prove that probabilities are calibrated. The built-in benchmark
does not establish real-world generalization, fairness, or clinical
suitability.

The dataset is used only for an educational classification demo.

## Portfolio Summary

Built a lightweight ML evaluation API with reproducible metrics,
input validation, confidence-based error investigation, and
interactive documentation.
