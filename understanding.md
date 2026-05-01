# Understanding This Fake News Detection Project

This file explains how your complete project works: dataset flow, model logic, backend API, frontend flow, current limitations, and what you should do next.

## 1. What This Project Is

This is an end-to-end fake news detection application built around the LIAR dataset.

It has three main parts:

1. **Machine learning backend**
   - Loads LIAR dataset TSV files.
   - Converts the original truth labels into binary labels.
   - Extracts text and metadata features.
   - Trains three ML models.
   - Saves trained model artifacts using `joblib`.

2. **Flask API**
   - Loads the trained model artifacts.
   - Accepts a claim from the frontend.
   - Converts the claim into the same feature format used during training.
   - Returns `REAL` or `FAKE`, confidence, and explanation data.

3. **React frontend**
   - Provides a user interface where a user enters a statement.
   - Sends the statement to the Flask API.
   - Displays the prediction, confidence score, credibility score, and influential keywords.

The main purpose of the project is not just to train a model in a notebook. It is structured like a real ML application: data, training, model persistence, API serving, frontend display, tests, and documentation.

## 2. Important Files

### Backend

- `backend/train.py`
  - Command-line entry point for training models.
  - It creates a `LiarModelTrainer` and calls `train_and_persist()`.

- `backend/model_service.py`
  - Most important ML file.
  - Contains the training class: `LiarModelTrainer`.
  - Contains the prediction service: `FakeNewsDetectionService`.
  - Handles model loading, training if artifacts are missing, prediction, confidence, and explanation logic.

- `backend/features.py`
  - Contains feature engineering logic.
  - Handles tokenization, TF-IDF vectorizer creation, speaker credibility, text statistics, party encoding, subject vectorization, and feature combination.

- `backend/data_utils.py`
  - Loads LIAR TSV files.
  - Defines the LIAR column names.
  - Converts six original LIAR labels into binary labels.

- `backend/app.py`
  - Flask API application.
  - Defines `/api/health`, `/api/stats`, `/api/models`, `/api/predict`, and `/api/feedback`.

- `backend/config.py`
  - Central configuration file.
  - Defines model paths, dataset paths, model names, deployed model, Flask port, input limits, and allowed frontend origins.

- `backend/models/training_summary.json`
  - Stores the latest training metadata and metrics.

### Frontend

- `frontend/src/pages/DetectorPage.jsx`
  - Main user page where users submit a claim.
  - Sends text, speaker, and party to the backend.

- `frontend/src/api/client.js`
  - Axios API client.
  - Calls `/api/predict`, `/api/health`, and `/api/stats`.

- `frontend/src/components/ResultPanel.jsx`
  - Displays the verdict, confidence, credibility score, and top keywords.

### Tests

- `tests/test_api.py`
  - Tests the Flask API routes using a fake model service.
  - Checks health, models, prediction validation, long text truncation, and feedback.

## 3. Dataset Logic

The project uses the LIAR dataset.

The expected files are:

```text
data/train.tsv
data/valid.tsv
data/test.tsv
```

The dataset is loaded in `backend/data_utils.py`.

Each row has these columns:

```text
id
label
statement
subject
speaker
job
state
party
barely_true_counts
false_counts
half_true_counts
mostly_true_counts
pants_on_fire_counts
context
```

## 4. Label Conversion Logic

The original LIAR dataset has six labels:

```text
pants-fire
false
barely-true
half-true
mostly-true
true
```

Your project converts them into a binary classification problem.

This mapping is defined in `LABEL_MAP` inside `backend/data_utils.py`:

```text
pants-fire   -> 0 -> FAKE
false        -> 0 -> FAKE
barely-true  -> 0 -> FAKE
half-true    -> 1 -> REAL
mostly-true  -> 1 -> REAL
true         -> 1 -> REAL
```

So internally:

- `0` means `FAKE`
- `1` means `REAL`

This is an important design decision. The model is not solving the full six-class LIAR task. It is solving a simpler binary version.

## 5. Data Cleaning Logic

When a TSV file is loaded:

1. The label is mapped from text into `0` or `1`.
2. Rows with missing labels or statements are removed.
3. Count columns are converted to numeric values.
4. Missing numeric values become `0.0`.
5. Text columns are converted to strings.
6. Empty text metadata becomes `"unknown"`.

This makes sure the model receives consistent input.

## 6. Text Preprocessing Logic

Text preprocessing happens in `tokenize_statement()` inside `backend/features.py`.

For each statement:

1. Convert the input into a string.
2. Split text using NLTK `wordpunct_tokenize`.
3. Lowercase each token.
4. Keep only alphabetic words.
5. Remove English stopwords.
6. Apply Snowball stemming.
7. Drop very short stemmed tokens.

Example idea:

```text
"Taxes are increasing rapidly!"
```

may become tokens like:

```text
tax
increas
rapid
```

The model does not directly understand grammar or factual evidence. It learns patterns from processed words and metadata.

## 7. TF-IDF Logic

The project uses `TfidfVectorizer`.

Configuration:

```python
TfidfVectorizer(
    tokenizer=tokenize_statement,
    token_pattern=None,
    lowercase=False,
    max_features=10000,
    ngram_range=(1, 2),
)
```

Meaning:

- `tokenizer=tokenize_statement`
  - Uses your custom tokenizer.

- `max_features=10000`
  - Keeps the top 10,000 most useful word or phrase features.

- `ngram_range=(1, 2)`
  - Uses single words and two-word phrases.
  - Example: `tax`, `health`, `tax cut`, `health care`.

TF-IDF gives higher weight to words that are important in one statement but not too common across all statements.

## 8. Metadata Feature Logic

The Random Forest model uses extra features beyond TF-IDF.

### Speaker Credibility

Speaker credibility is calculated from the speaker's historical LIAR counts.

The formula is:

```text
total_history =
  barely_true_counts
+ false_counts
+ half_true_counts
+ mostly_true_counts
+ pants_on_fire_counts

false_penalty =
  (false_counts + pants_on_fire_counts) / total_history

speaker_credibility =
  1.0 - false_penalty
```

Then the value is clipped between `0.0` and `1.0`.

If a speaker has no history, credibility becomes `0.5`.

Interpretation:

- High credibility means the speaker historically had fewer false and pants-fire claims.
- Low credibility means the speaker historically had more false and pants-fire claims.

Important: this does not prove a new claim is true or false. It is only a historical signal.

### Text Statistics

The model also calculates:

- `text_length`
  - Number of characters in the claim.

- `punctuation_count`
  - Number of punctuation characters.

- `all_caps_ratio`
  - Ratio of uppercase words to alphabetic words.

- `exclamation_count`
  - Number of `!` characters.

These are simple style features.

### Party Encoding

Political party is encoded using `OneHotEncoder`.

Example:

```text
democrat    -> party_democrat = 1
republican  -> party_republican = 1
unknown     -> party_unknown = 1
```

Unknown parties are ignored safely because `handle_unknown="ignore"` is used.

### Subject Vectorization

Subjects are split by comma and encoded using `CountVectorizer`.

Example:

```text
"health-care,taxes"
```

becomes binary subject tags like:

```text
health-care = 1
taxes = 1
```

The backend supports `subject`, but the current frontend does not send it.

## 9. Final Feature Matrix

For the Random Forest model, all features are joined together:

```text
full_features =
[
  TF-IDF text features,
  numeric metadata features,
  party one-hot features,
  subject binary features
]
```

This combination happens in `combine_engineered_features()` using `scipy.sparse.hstack`.

The result is a large sparse matrix that the model can train on.

## 10. Models Used

The training pipeline trains three models.

### 1. Logistic Regression

File:

```text
backend/models/logistic_regression.pkl
```

Features used:

```text
TF-IDF only
```

Why it is used:

- Strong baseline for text classification.
- Works well with sparse TF-IDF features.
- More interpretable than many models.

Configuration:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)
```

### 2. Naive Bayes

File:

```text
backend/models/naive_bayes.pkl
```

Features used:

```text
TF-IDF only
```

Why it is used:

- Very fast text classification baseline.
- Commonly used in spam/news classification.
- Assumes features are conditionally independent, which is not fully true, but often works decently for text.

Configuration:

```python
MultinomialNB(alpha=0.5)
```

### 3. Random Forest

File:

```text
backend/models/random_forest.pkl
```

Features used:

```text
TF-IDF + speaker credibility + text statistics + party + subject
```

Why it is used:

- Can combine text features and metadata features.
- Uses many decision trees.
- Each tree votes, and the forest combines the results.

Configuration:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=1,
    random_state=42,
)
```

This is the currently deployed model because `DEPLOYED_MODEL_NAME` defaults to:

```text
RandomForest
```

## 11. Training Flow

Training starts from:

```bash
cd backend
python train.py --data-dir ../data
```

Flow:

1. `train.py` creates `LiarModelTrainer`.
2. `LiarModelTrainer.train_and_persist()` finds the dataset directory.
3. It loads `train`, `valid`, and `test`.
4. It builds speaker credibility scores from the training split.
5. It trains Logistic Regression on TF-IDF.
6. It trains Naive Bayes on TF-IDF.
7. It builds engineered features for Random Forest.
8. It trains Random Forest.
9. It evaluates all models on validation and test sets.
10. It saves model artifacts into `backend/models`.
11. It writes metrics into `backend/models/training_summary.json`.

## 12. Current Training Result Warning

Your current `backend/models/training_summary.json` says:

```json
{
  "train": 16,
  "valid": 8,
  "test": 8
}
```

This means the saved models were trained on a very tiny dataset.

The metrics show 100% accuracy, precision, recall, F1, and ROC-AUC, but those numbers are not reliable because the dataset is too small.

This is probably a sample or placeholder dataset, not the full LIAR dataset.

For a real project report or demo, you should train using the full LIAR dataset:

```text
train: around 10,000 rows
valid: around 1,200 rows
test: around 1,200 rows
```

Until you retrain on the full dataset, you should not claim that the model has real 100% accuracy.

## 13. Prediction Flow

When the user enters a claim in the frontend:

1. User types text in `DetectorPage.jsx`.
2. Frontend validates that text is not empty and is at least 10 characters.
3. Frontend sends this payload:

```json
{
  "text": "claim text",
  "speaker": "optional speaker",
  "party": "selected party"
}
```

4. Axios sends it to:

```text
POST http://localhost:5000/api/predict
```

5. Flask receives the request in `backend/app.py`.
6. `FakeNewsDetectionService.predict()` validates the payload.
7. The same vectorizers and encoders from training transform the new input.
8. The deployed model runs:

```python
predict_proba(feature_matrix)
```

9. The class with the highest probability is selected.
10. If predicted class is `1`, API returns `REAL`.
11. If predicted class is `0`, API returns `FAKE`.
12. The response is displayed in `ResultPanel.jsx`.

## 14. Confidence Logic

The model returns probabilities for both classes.

Example:

```text
FAKE probability = 0.22
REAL probability = 0.78
```

The highest probability wins.

In this example:

```text
prediction = REAL
confidence = 0.78
```

Important: confidence is model confidence, not factual certainty.

A confidence of `0.78` does not mean the claim is objectively 78% true. It means the model assigned 78% probability to the selected class based on learned patterns.

## 15. Explanation Logic

The API returns:

```json
{
  "top_tfidf_features": [],
  "explanation_method": "...",
  "credibility_score": 0.61,
  "text_statistics": {}
}
```

### For Logistic Regression

Explanation uses:

```text
active TF-IDF value * model coefficient
```

This shows which words pushed the prediction toward the selected class.

### For Naive Bayes

Explanation uses:

```text
active TF-IDF value * difference between class log probabilities
```

This shows which words are more associated with one class than the other.

### For Random Forest

Explanation uses:

```text
active TF-IDF value * global feature importance
```

This is approximate.

Random Forest does not naturally provide exact local word-level explanations without extra libraries like SHAP or LIME. So your explanation shows important words that were active in the statement, weighted by global Random Forest feature importance.

## 16. API Routes

### `/api/health`

Checks whether the backend and model service are ready.

Returns:

- status
- deployed model
- available models
- loaded model count
- startup error if any

### `/api/stats`

Returns training metrics from `training_summary.json`.

### `/api/models`

Returns model artifact information:

- model name
- file name
- whether it loaded
- file size
- path

### `/api/predict`

Main prediction endpoint.

Accepts:

```json
{
  "text": "required",
  "speaker": "optional",
  "party": "optional",
  "subject": "optional"
}
```

Returns:

```json
{
  "prediction_id": "...",
  "prediction": "REAL or FAKE",
  "confidence": 0.8732,
  "model": "RandomForest",
  "explanation": {}
}
```

### `/api/feedback`

Accepts feedback about a prediction:

```json
{
  "prediction_id": "...",
  "correct": true
}
```

Currently it saves feedback into:

```text
backend/feedback.json
```

## 17. Frontend Logic

The frontend is a React app.

Main flow:

1. `DetectorPage.jsx` stores form state:

```js
{
  text: "",
  speaker: "",
  party: "unknown"
}
```

2. On submit, it validates the text.
3. It calls `predictNews(payload)`.
4. `predictNews()` uses Axios to call `/api/predict`.
5. Response is stored in React state.
6. `ResultPanel.jsx` displays:
   - prediction
   - confidence
   - primary model name
   - speaker credibility
   - top influential keywords

The frontend currently does not send `subject`, even though the backend supports it.

## 18. Tests

The test file `tests/test_api.py` tests backend routes using a fake service.

It checks:

- `/api/health` returns 200.
- `/api/models` returns loaded model information.
- `/api/predict` returns a valid prediction for valid input.
- Short text returns 400.
- Empty text returns 400.
- Very long text is truncated to 5000 characters.
- `/api/feedback` saves boolean feedback.

These are API behavior tests. They do not test real model accuracy.

## 19. Strengths Of Your Project

Your project has a good full-stack ML structure:

- Separate training pipeline.
- Reusable feature engineering code.
- Persisted model artifacts.
- Flask API for inference.
- React frontend.
- Health and stats routes.
- Basic rate limiting.
- CORS configuration.
- Feedback endpoint.
- Tests for API behavior.
- Clear architecture documentation.

This is better than a notebook-only ML project because it shows deployment thinking.

## 20. Current Limitations

### 1. Current dataset is too small

The current model artifacts were trained on only:

```text
16 train rows
8 validation rows
8 test rows
```

So the current 100% score is not meaningful.

### 2. The model does not verify facts externally

It predicts based on language and metadata patterns. It does not search the web, retrieve evidence, or compare claims against trusted sources.

### 3. Binary mapping loses detail

The original LIAR labels are more nuanced. Converting six labels into two classes makes the system easier to use, but loses truthfulness detail.

### 4. Random Forest explanation is approximate

The top keywords are not a full causal explanation.

### 5. Frontend does not collect subject

The backend supports subject, but the UI sends only text, speaker, and party.

### 6. Feedback is saved locally

Feedback goes into `backend/feedback.json`. For production, this should move to a database.

## 21. What You Should Do Next

### Step 1: Download and use the full LIAR dataset

Your most important next step is to replace the tiny sample data with the real LIAR files.

You need:

```text
data/train.tsv
data/valid.tsv
data/test.tsv
```

Then retrain:

```bash
cd backend
python train.py --data-dir ../data
```

After retraining, check:

```text
backend/models/training_summary.json
```

The dataset sizes should be much larger than 16, 8, and 8.

### Step 2: Update README metrics

Your `README.md` still has placeholder model performance values.

After training on full data, copy real metrics from:

```text
backend/models/training_summary.json
```

Update the model performance table.

### Step 3: Add subject input to the frontend

The backend supports `subject`, but the frontend does not send it.

Add a subject field in:

```text
frontend/src/pages/DetectorPage.jsx
```

Then include it in the payload:

```js
{
  text: form.text,
  speaker: form.speaker || undefined,
  party: form.party || undefined,
  subject: form.subject || undefined
}
```

This allows the Random Forest model to use subject features during real predictions.

### Step 4: Run tests

Run backend tests:

```bash
python -m pytest tests
```

Run frontend tests:

```bash
cd frontend
npm test
```

Build frontend:

```bash
cd frontend
npm run build
```

### Step 5: Improve model evaluation

Add more serious evaluation:

- Confusion matrix visualization.
- Class distribution.
- Precision and recall per class.
- ROC curve.
- Calibration curve.
- Error examples where model failed.

This will help you explain the model better in a report or viva.

### Step 6: Try stronger models

After your classical ML pipeline works properly, try:

- Linear SVM with TF-IDF.
- Gradient Boosting or XGBoost on engineered features.
- BERT or RoBERTa fine-tuning.
- Sentence-transformer embeddings plus classifier.

For fake news detection, transformer models usually understand claim semantics better than TF-IDF.

### Step 7: Add evidence retrieval

A real fake news system should not only classify from text. It should retrieve evidence.

Future version idea:

1. User enters claim.
2. System searches trusted sources.
3. System retrieves related evidence.
4. Model compares claim with evidence.
5. UI shows evidence links and verdict.

This would make the project closer to real fact-checking.

### Step 8: Improve production readiness

Before deploying publicly:

- Replace `feedback.json` with a database.
- Add model versioning.
- Add structured logs.
- Add request IDs.
- Add stronger abuse/rate limiting.
- Add user-facing disclaimers.
- Deploy Flask with Gunicorn.
- Deploy React build on Vercel, Netlify, or similar.

## 22. Simple Explanation You Can Say In Presentation

This project uses the LIAR dataset to classify political statements as fake or real. The original six truth labels are converted into two classes. The text is cleaned, tokenized, stemmed, and converted into TF-IDF features. The project trains Logistic Regression and Naive Bayes as text-only baselines, and a Random Forest model that also uses metadata such as speaker credibility, political party, subject, and text style statistics. The trained model is saved and served through a Flask API. The React frontend sends user claims to the API and displays the prediction, confidence score, speaker credibility, and influential keywords. The current saved model was trained on a very small sample dataset, so the next important step is to train on the full LIAR dataset and update the reported metrics.

## 23. Final Key Point

The main logic of your project is:

```text
LIAR data
-> clean labels and metadata
-> convert text into TF-IDF
-> calculate speaker/text/party/subject features
-> train ML models
-> save model artifacts
-> load model in Flask
-> transform user input the same way
-> predict REAL or FAKE
-> show result in React
```

The most urgent next action is to train on the full LIAR dataset, because the current trained model is based on only 32 total rows across train, validation, and test.
