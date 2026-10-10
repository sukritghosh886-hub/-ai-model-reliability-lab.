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


# AI Model Reliability Lab

A lightweight, browser-based tool for evaluating binary classification predictions.

**Created by:** Sukrit Ghosh

## Features

- Accuracy, precision, recall, and F1 score
- Confusion matrix: true negatives, false positives, false negatives, and true positives
- Optional mean prediction latency from supplied measurements
- CSV upload and downloadable sample data
- Responsive interface for mobile and desktop browsers
- Local browser processing without sending uploaded CSV data to an application server

## How to use

1. Open the deployed app.
2. Download `sample_predictions.csv`, or prepare your own CSV.
3. Include the required columns `y_true` and `y_pred`.
4. Use only binary labels: `0` and `1`.
5. Optionally include `latency_ms` with non-negative numerical measurements.
6. Upload the CSV and select **Evaluate CSV**.

## Sample CSV format

    y_true,y_pred,latency_ms
    1,1,12
    1,0,15
    0,0,10
    0,0,11
    1,1,13
    0,1,17

## Technology

- HTML
- CSS
- JavaScript
- Browser File API

No Python installation, model downloads, GPU, or backend server is required.

## Validation

The included six-row sample should produce:

- Accuracy: 66.67%
- Precision: 66.67%
- Recall: 66.67%
- F1 score: 66.67%
- Confusion matrix: TN=2, FP=1, FN=1, TP=2
- Mean supplied latency: 13.00 ms

## Limitations

This is an educational evaluation utility, not a production AI certification system. It evaluates supplied labels and measurements. It does not train models, independently verify predictions, establish causation, or assess model fairness, security, robustness, or real-world reliability.

The CSV parser supports simple comma-separated files, not every quoted-field format.

## License

MIT


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
