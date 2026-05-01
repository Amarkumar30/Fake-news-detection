# Working Process Of The Fake News Detection Application

This document explains the complete working process of the application in a viva-friendly way. It focuses on what the app does, why the selected algorithms and features are used, how the prediction is generated, and what every result shown on the frontend means.

## 1. Basic Idea Of The Application

The application checks whether a political claim or news statement looks more like `REAL` or `FAKE` according to patterns learned from the LIAR dataset.

The system does not manually fact-check the statement from the internet. It does not search Google or compare the statement with live evidence. Instead, it uses a trained machine learning model. The model has learned from thousands of previously labelled political statements.

In simple words:

```text
User enters a claim
-> app sends claim to backend
-> backend converts claim into numerical features
-> trained ML model predicts REAL or FAKE
-> app shows verdict, confidence, credibility, and important keywords
```

## 2. Current Dataset Used After Retraining

After retraining, the model summary shows that the project is now using a much larger dataset:

```text
Training rows:   10240
Validation rows: 1284
Test rows:       1267
```

This is much better than the earlier small sample dataset. Now the model performance is more realistic.

The current deployed model is:

```text
RandomForest
```

The best validation model is also:

```text
RandomForest
```

## 3. Current Model Performance

The current metrics are stored in:

```text
backend/models/training_summary.json
```

### Logistic Regression

Validation:

```text
Accuracy:  0.6083
Precision: 0.6295
Recall:    0.6003
F1 Score:  0.6146
ROC-AUC:   0.6587
```

Test:

```text
Accuracy:  0.6062
Precision: 0.6516
Recall:    0.6471
F1 Score:  0.6493
ROC-AUC:   0.6518
```

### Naive Bayes

Validation:

```text
Accuracy:  0.5989
Precision: 0.5897
Recall:    0.7530
F1 Score:  0.6614
ROC-AUC:   0.6467
```

Test:

```text
Accuracy:  0.6069
Precision: 0.6205
Recall:    0.7787
F1 Score:  0.6907
ROC-AUC:   0.6494
```

### Random Forest

Validation:

```text
Accuracy:  0.6550
Precision: 0.6455
Recall:    0.7470
F1 Score:  0.6926
ROC-AUC:   0.7163
```

Test:

```text
Accuracy:  0.6669
Precision: 0.6825
Recall:    0.7647
F1 Score:  0.7213
ROC-AUC:   0.7036
```

Random Forest is selected because it gives the best validation F1 score and also the strongest test F1 score among the three models.

## 4. Why The Accuracy Is Around 60-67 Percent

This is normal for this type of project.

The LIAR dataset is difficult because:

1. Claims are short.
2. Truth often depends on external facts.
3. Many statements are political and subjective.
4. The original labels are not just true/false; they have six levels.
5. The project converts six labels into two labels, which loses some detail.

So if a professor asks why accuracy is not 90 percent, the correct answer is:

The model is using classical machine learning with TF-IDF and metadata features. It learns linguistic and metadata patterns, but it does not verify facts using external evidence. Fake news detection is hard because truth often depends on real-world context, not only wording.

## 5. Original Labels And Binary Conversion

The LIAR dataset has six original labels:

```text
pants-fire
false
barely-true
half-true
mostly-true
true
```

The project converts them into two classes:

```text
pants-fire   -> FAKE -> 0
false        -> FAKE -> 0
barely-true  -> FAKE -> 0
half-true    -> REAL -> 1
mostly-true  -> REAL -> 1
true         -> REAL -> 1
```

Why this is done:

The frontend gives a simple verdict to the user: `REAL` or `FAKE`. A binary output is easier to understand than six truthfulness levels.

Limitation:

`barely-true` and `half-true` are close in meaning, but they go into different binary classes. This can make the task noisy.

## 6. Features Used In The Model

Your professor said to use fewer features and algorithms, but know them properly. This project mainly uses a small, explainable set of features:

1. TF-IDF text features
2. Speaker credibility
3. Text statistics
4. Political party
5. Subject tags

These features are enough to explain in viva without making the system too complex.

## 7. Feature 1: TF-IDF Text Features

TF-IDF means:

```text
Term Frequency - Inverse Document Frequency
```

It converts text into numbers.

### Term Frequency

Term frequency checks how often a word appears in a statement.

Example:

```text
"unemployment fell because unemployment policies changed"
```

The word `unemployment` appears more than once, so it gets more importance in that statement.

### Inverse Document Frequency

Inverse document frequency reduces the value of very common words.

Words like:

```text
the
is
are
of
and
```

are not very useful because they appear almost everywhere.

Words like:

```text
unemployment
budget
crime
tax
health
```

can be more meaningful.

### Why TF-IDF Is Used

TF-IDF is used because:

1. It is simple.
2. It works well for text classification.
3. It is easy to explain.
4. It creates numerical input for machine learning models.
5. It helps identify important words and phrases.

The project uses:

```python
max_features=10000
ngram_range=(1, 2)
```

Meaning:

- It keeps up to 10,000 text features.
- It uses single words and two-word phrases.

Example:

```text
Single word: unemployment
Two-word phrase: health care
```

## 8. Text Preprocessing Before TF-IDF

Before TF-IDF, the text is cleaned.

The project does this:

1. Tokenizes the statement into words.
2. Converts words to lowercase.
3. Removes punctuation and non-alphabetic tokens.
4. Removes English stopwords.
5. Applies stemming.

Example input:

```text
"The unemployment rate has fallen to its lowest level in fifty years!"
```

After preprocessing, important tokens may look like:

```text
unemploy
rate
fallen
lowest
level
fifti
year
```

Stemming converts words to root-like forms:

```text
unemployment -> unemploy
increasing   -> increas
years        -> year
```

Why stemming is used:

It reduces similar forms of a word into one common form, so the model does not treat `year` and `years` as completely different concepts.

## 9. Feature 2: Speaker Credibility

The LIAR dataset contains historical count values for each speaker:

```text
barely_true_counts
false_counts
half_true_counts
mostly_true_counts
pants_on_fire_counts
```

The project uses these counts to calculate a speaker credibility score.

Formula:

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

If no history exists:

```text
speaker_credibility = 0.5
```

Why this feature is used:

Some speakers historically make more false claims, and some speakers historically make more accurate claims. This does not prove whether the new claim is true, but it gives useful context.

Important viva answer:

Speaker credibility is not final proof. It is only a supporting signal. A person with good history can still make a false claim, and a person with poor history can still make a true claim.

## 10. Feature 3: Text Statistics

The project calculates simple statistics from the input text:

```text
text_length
punctuation_count
all_caps_ratio
exclamation_count
```

### text_length

Number of characters in the statement.

### punctuation_count

Total punctuation characters.

### all_caps_ratio

Ratio of uppercase words compared with total alphabetic words.

Example:

```text
"THIS IS FAKE NEWS"
```

has a high all-caps ratio.

### exclamation_count

Number of exclamation marks.

Why these are used:

Some misleading or viral statements may use emotional formatting, excessive punctuation, or exaggerated style. These features are not enough alone, but they can help the model.

## 11. Feature 4: Political Party

The frontend sends a political party value:

```text
unknown
democrat
republican
independent
libertarian
green
constitution
nonpartisan
```

The backend converts the party into numerical form using one-hot encoding.

Example:

```text
party = democrat
```

becomes:

```text
party_democrat = 1
party_republican = 0
party_unknown = 0
...
```

Why this is used:

The LIAR dataset is political, so party can be a useful metadata signal. But it must be handled carefully because party should not be treated as proof of truth or falsehood.

## 12. Feature 5: Subject Tags

The backend supports subject tags such as:

```text
economy
taxes
health-care
crime
education
```

Subjects are converted into binary features.

Example:

```text
"economy,taxes"
```

becomes:

```text
economy = 1
taxes = 1
```

Current frontend note:

The backend supports `subject`, but the current frontend does not ask the user to enter subject. So during normal app use, subject usually becomes `unknown`.

## 13. Algorithms Used

The project trains three algorithms:

1. Logistic Regression
2. Naive Bayes
3. Random Forest

This is a reasonable number of algorithms for a college project because all three are standard, explainable classical ML models.

## 14. Algorithm 1: Logistic Regression

Logistic Regression is a linear classification algorithm.

It tries to draw a decision boundary between classes.

In this project:

```text
Input:  TF-IDF text features
Output: probability of FAKE or REAL
```

Why it is used:

1. It is a strong baseline for text classification.
2. It works well with sparse TF-IDF features.
3. It is interpretable.
4. It gives probabilities using `predict_proba()`.

How it works simply:

Each word feature gets a weight. Some words push the model toward `REAL`, and some words push it toward `FAKE`. The model combines all feature weights and passes the result through a logistic function to produce a probability.

## 15. Algorithm 2: Naive Bayes

Naive Bayes is a probabilistic classifier.

In this project:

```text
Input:  TF-IDF text features
Output: probability of FAKE or REAL
```

Why it is used:

1. It is simple and fast.
2. It is commonly used for text classification.
3. It works well for word-count-like features.
4. It gives a useful comparison against Logistic Regression.

Why it is called naive:

It assumes that features are independent of each other. In text, this assumption is not fully true because words are related, but Naive Bayes can still perform reasonably well.

## 16. Algorithm 3: Random Forest

Random Forest is the deployed model.

In this project:

```text
Input:  TF-IDF + speaker credibility + text statistics + party + subject
Output: probability of FAKE or REAL
```

Why it is used:

1. It can use both text features and metadata features.
2. It combines many decision trees.
3. It reduces the weakness of a single decision tree.
4. It performed best after retraining.

How it works simply:

Random Forest builds many decision trees. Each tree makes a prediction. The forest combines all tree outputs and gives the final class probability.

Configuration used:

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

Meaning:

- `n_estimators=300`
  - The forest has 300 trees.

- `max_depth=30`
  - Each tree can grow up to depth 30.

- `min_samples_leaf=2`
  - A leaf node must have at least 2 samples.

- `class_weight="balanced_subsample"`
  - Helps handle class imbalance inside tree samples.

- `random_state=42`
  - Makes training reproducible.

## 17. Why Random Forest Was Selected

Random Forest is selected because it performs best on validation F1 score.

Validation F1 scores:

```text
Logistic Regression: 0.6146
Naive Bayes:         0.6614
Random Forest:       0.6926
```

Test F1 scores:

```text
Logistic Regression: 0.6493
Naive Bayes:         0.6907
Random Forest:       0.7213
```

F1 score is important because it balances precision and recall.

For fake news detection, only accuracy is not enough. If one class is more common, accuracy can look good even when the model is weak. F1 gives a better view of classification quality.

## 18. Complete Training Process

Training starts with:

```bash
cd backend
python train.py --data-dir ../data
```

The process:

1. `train.py` starts the training process.
2. It creates `LiarModelTrainer`.
3. The trainer loads `train.tsv`, `valid.tsv`, and `test.tsv`.
4. Labels are converted from six classes to binary labels.
5. Text columns are cleaned.
6. Numeric history columns are converted to numbers.
7. Speaker credibility values are calculated from the training data.
8. TF-IDF vectorizer is fitted on training statements.
9. Logistic Regression is trained on TF-IDF features.
10. Naive Bayes is trained on TF-IDF features.
11. A second TF-IDF vectorizer is fitted for engineered features.
12. Party encoder is fitted.
13. Subject vectorizer is fitted.
14. Numeric feature matrix is created.
15. TF-IDF, numeric, party, and subject features are joined.
16. Random Forest is trained on the combined feature matrix.
17. All models are evaluated on validation and test data.
18. Artifacts are saved into `backend/models`.
19. Metrics are saved into `training_summary.json`.

## 19. Files Created After Training

After training, these files exist:

```text
backend/models/logistic_regression.pkl
backend/models/naive_bayes.pkl
backend/models/random_forest.pkl
backend/models/training_summary.json
```

Each `.pkl` file stores:

- trained model
- vectorizer
- encoders
- feature names
- speaker credibility lookup
- default speaker score

This is important because prediction must use the same transformation logic as training.

## 20. Application Runtime Flow

When the application runs:

1. Flask starts from `backend/app.py`.
2. `FakeNewsDetectionService` is created.
3. It checks whether model files already exist.
4. If model files exist, it loads them.
5. If model files do not exist, it tries to train models.
6. React frontend starts separately.
7. Frontend checks backend health.
8. User enters a claim.
9. Frontend sends request to backend.
10. Backend validates the request.
11. Backend transforms the input into model features.
12. Random Forest predicts probabilities.
13. Backend returns JSON result.
14. Frontend displays the result.

## 21. Frontend Screen Before Prediction

Before prediction, the app shows:

### Header/Hero Section

It shows:

```text
Forensic Claim Analysis
IS IT REAL?
```

It also describes that the detector combines linguistic signals, speaker credibility, and LIAR metadata.

### System Focus Panel

It shows:

```text
Dataset: LIAR, 12,836 labeled statements
Deployment: Flask API at localhost:5000
Decisioning: Binary REAL / FAKE confidence output
```

### Detector Console

This is the input form.

It contains:

1. Claim or news excerpt textarea
2. Character count
3. Recommended character range
4. Speaker name input
5. Political party dropdown
6. Analyze button
7. Reset button

### Result Panel Placeholder

Before submitting, the result panel says:

```text
Awaiting Analysis
Submit a claim to see the forensic breakdown.
```

It also says the detector will return:

- binary verdict
- confidence score
- historical speaker credibility
- top weighted keywords

### Example Cards

The app also shows sample statements that can be clicked to fill the form.

Examples include:

- Campaign Ad Claim
- Budget Talking Point
- Debate Soundbite
- Viral Social Post

## 22. Input Validation In The Frontend

The frontend checks:

1. Text must not be empty.
2. Text must be at least 10 characters.
3. Text can go up to 5000 characters.

If the text is empty:

```text
Text cannot be empty
```

If the text is too short:

```text
Please enter a longer statement
```

The frontend also shows character count:

```text
0 characters
Recommended: 50-5000
```

## 23. API Request Sent By Frontend

When the user clicks `Analyze`, the frontend sends:

```json
{
  "text": "The unemployment rate has fallen to its lowest level in fifty years according to the latest government report.",
  "speaker": "government official",
  "party": "democrat"
}
```

This goes to:

```text
POST /api/predict
```

Full local URL:

```text
http://localhost:5000/api/predict
```

## 24. Backend Validation

The backend validates the payload.

It checks:

1. Request body must be JSON.
2. `text` must exist.
3. `text` must not be empty.
4. `text` must be at least 10 characters.
5. If text is longer than 5000 characters, it is truncated.
6. Missing speaker becomes `unknown`.
7. Missing party becomes `unknown`.
8. Missing subject becomes `unknown`.

## 25. Backend Feature Creation During Prediction

For a new statement, the backend creates the same type of features used during training.

For Random Forest:

```text
New text
-> TF-IDF vector
-> speaker credibility score
-> text statistics
-> party one-hot vector
-> subject vector
-> combined feature matrix
-> Random Forest prediction
```

The important rule is:

The backend does not fit a new vectorizer during prediction. It uses the saved vectorizer from training. This ensures the feature columns are the same as during training.

## 26. Example Prediction

Example input:

```text
Claim:
The unemployment rate has fallen to its lowest level in fifty years according to the latest government report.

Speaker:
government official

Party:
democrat
```

Actual prediction from the current trained model:

```json
{
  "prediction": "REAL",
  "confidence": 0.585,
  "model": "RandomForest",
  "explanation": {
    "top_tfidf_features": [
      {
        "feature": "rate",
        "contribution": 0.001
      },
      {
        "feature": "year",
        "contribution": 0.0007
      },
      {
        "feature": "govern",
        "contribution": 0.0005
      },
      {
        "feature": "lowest",
        "contribution": 0.0004
      },
      {
        "feature": "unemploy",
        "contribution": 0.0002
      }
    ],
    "explanation_method": "importance_weighted_tfidf",
    "credibility_score": 0.6793,
    "text_statistics": {
      "text_length": 110.0,
      "punctuation_count": 1.0,
      "all_caps_ratio": 0.0,
      "exclamation_count": 0.0
    }
  }
}
```

## 27. How To Explain This Example In Viva

The model predicted:

```text
REAL
```

with confidence:

```text
58.5%
```

This means Random Forest assigned the highest probability to the `REAL` class.

It does not mean the statement is guaranteed to be true. It means that according to the learned patterns from the LIAR dataset, the statement is closer to the `REAL` class than the `FAKE` class.

The confidence is not very high. A 58.5% confidence means the model is only moderately sure.

The speaker credibility is:

```text
0.6793
```

The frontend shows this approximately as:

```text
68 / 100
```

The top influential keywords are:

```text
rate
year
govern
lowest
unemploy
```

These are stemmed or processed terms from the statement. For example:

```text
government -> govern
unemployment -> unemploy
```

## 28. What The App Shows After Prediction

After prediction, the result panel shows several things.

### 1. Verdict

Example:

```text
REAL
```

This comes from:

```python
prediction = "REAL" if predicted_index == 1 else "FAKE"
```

If the model selects class `1`, the app shows `REAL`.

If the model selects class `0`, the app shows `FAKE`.

### 2. Primary Model

Example:

```text
Primary model: RandomForest
```

This tells the user which trained model produced the result.

### 3. Confidence Ring

Example:

```text
Confidence 58%
```

The frontend converts:

```text
0.585 -> 58.5% -> displayed around 58%
```

This appears as a circular animated confidence display.

### 4. Speaker Credibility Gauge

Example:

```text
Speaker Credibility
68 / 100
```

This is based on the speaker history score calculated from LIAR metadata.

If no speaker is supplied, the app shows:

```text
Unavailable
```

### 5. Analyst Note

For the example, the app builds a sentence like:

```text
REAL at 59% confidence. Speaker history scored 68 out of 100 credibility. The strongest textual signal in this pass was "rate".
```

This note is generated in `ResultPanel.jsx`.

It summarizes:

- verdict
- confidence
- speaker credibility
- strongest keyword

### 6. Top Influential Keywords

The app shows keyword chips.

For the example:

```text
rate       0.001
year       0.0007
govern     0.0005
lowest     0.0004
unemploy   0.0002
```

These values are not percentages. They are contribution scores calculated from TF-IDF values and feature importance.

### 7. Explanation Method

For Random Forest, the app shows:

```text
importance_weighted_tfidf
```

This means:

```text
active TF-IDF terms * Random Forest global feature importance
```

## 29. How Confidence Is Calculated

The model uses:

```python
predict_proba()
```

This returns probabilities for both classes.

Example:

```text
FAKE probability = 0.415
REAL probability = 0.585
```

The model chooses the larger one:

```text
REAL
```

Confidence becomes:

```text
0.585
```

Displayed on UI:

```text
58% or 59%
```

## 30. How Keyword Explanation Works

For Random Forest, exact local word explanation is difficult without extra explainability libraries.

So the project uses an approximate method:

```text
keyword contribution = TF-IDF value of keyword * global feature importance of keyword
```

This means:

1. The word must appear in the input.
2. The word must be recognized by the trained TF-IDF vectorizer.
3. The Random Forest must consider that feature important globally.
4. The top five positive contribution words are displayed.

Important viva answer:

The keyword explanation is not a perfect causal explanation. It is an interpretable approximation showing which active words had higher learned importance.

## 31. Confusion Matrix Meaning

For Random Forest test data, the confusion matrix is:

```text
[
  [299, 254],
  [168, 546]
]
```

This means:

```text
True FAKE predicted as FAKE: 299
True FAKE predicted as REAL: 254
True REAL predicted as FAKE: 168
True REAL predicted as REAL: 546
```

So the model is better at identifying `REAL` statements than `FAKE` statements in the current test set.

This matches the recall:

```text
Recall: 0.7647
```

The model correctly catches many real examples, but it still makes mistakes.

## 32. Meaning Of Evaluation Metrics

### Accuracy

Accuracy means:

```text
correct predictions / total predictions
```

Random Forest test accuracy:

```text
0.6669
```

So around 66.69% of test predictions were correct.

### Precision

Precision answers:

```text
When the model predicts REAL, how often is it actually REAL?
```

Random Forest test precision:

```text
0.6825
```

### Recall

Recall answers:

```text
Out of all actual REAL statements, how many did the model catch?
```

Random Forest test recall:

```text
0.7647
```

### F1 Score

F1 balances precision and recall.

Random Forest test F1:

```text
0.7213
```

This is why Random Forest is a good deployed choice in this project.

### ROC-AUC

ROC-AUC measures how well the model separates the two classes across different thresholds.

Random Forest test ROC-AUC:

```text
0.7036
```

A value above 0.5 means the model is better than random guessing.

## 33. Why We Use Validation And Test Data

Training data is used to teach the model.

Validation data is used to compare models and choose the best one.

Test data is used for final evaluation.

In this project:

```text
Random Forest had the best validation F1 score.
Therefore, it was selected as the deployed model.
```

## 34. What Happens If Backend Is Not Ready

The frontend checks backend health using:

```text
GET /api/health
```

If backend is ready, the app works normally.

If backend has a problem, the app can show a backend connection error.

The health route returns:

```text
status
deployed_model
available_models
loaded_model_count
startup_error
```

## 35. What Happens If User Enters Bad Input

### Empty Text

Frontend/backend returns:

```text
Text cannot be empty
```

### Too Short Text

Frontend/backend returns:

```text
Please enter a longer statement
```

### Too Long Text

Backend truncates it to 5000 characters.

### Too Many Requests

The backend has rate limiting:

```text
30 requests per minute
```

If exceeded:

```text
Rate limit exceeded. Please try again later.
```

## 36. Feedback Feature

The backend has:

```text
POST /api/feedback
```

It accepts:

```json
{
  "prediction_id": "some-id",
  "correct": true
}
```

Currently feedback is saved in:

```text
backend/feedback.json
```

This is useful for future improvement, but it is not currently used to retrain the model automatically.

## 37. Simple End-To-End Example For Viva

You can explain the application like this:

```text
First, the user enters a political claim in the React frontend.
The frontend validates the text and sends it to the Flask backend.
The backend loads the saved Random Forest model and the saved vectorizers.
The input text is tokenized, cleaned, stemmed, and converted into TF-IDF values.
The backend also calculates speaker credibility and simple text statistics.
Party is converted with one-hot encoding, and subject is handled as a tag feature.
All features are combined into one sparse matrix.
The Random Forest model calculates probabilities for FAKE and REAL.
The class with higher probability becomes the final verdict.
The frontend displays the verdict, confidence, model name, speaker credibility, analyst note, and top influential keywords.
```

## 38. Strong Viva Questions And Answers

### Why did you use TF-IDF?

TF-IDF is simple, interpretable, and effective for text classification. It converts text into numerical features by giving importance to words that are frequent in one document but not common across all documents.

### Why did you use Random Forest?

Random Forest can combine text features with metadata features such as speaker credibility, party, subject, and text statistics. It also performed best based on validation F1 score.

### Why not use only accuracy?

Accuracy can be misleading if classes are imbalanced. F1 score is better because it balances precision and recall.

### Why is the model not perfect?

Fake news detection requires real-world knowledge. This model only learns from text and metadata patterns. It does not verify facts from external sources.

### What is speaker credibility?

It is a score calculated from the speaker's historical LIAR counts. More false and pants-fire claims reduce credibility.

### Does speaker credibility prove a statement is false?

No. It is only a supporting feature. The current statement still needs text and context analysis.

### What is the limitation of binary labels?

The original dataset has six labels. Converting them into two labels makes the app easier to use but loses nuance.

### What does confidence mean?

Confidence is the probability assigned by the model to the selected class. It is not a guarantee of factual truth.

### What does `importance_weighted_tfidf` mean?

It means the explanation is calculated using active TF-IDF words weighted by Random Forest feature importance.

### Is this a real fact-checker?

No. It is a machine learning classifier and decision-support tool. A real fact-checker should also retrieve and compare external evidence.

## 39. Final Summary

The complete working process is:

```text
LIAR dataset
-> label conversion
-> text cleaning
-> TF-IDF feature extraction
-> metadata feature extraction
-> train Logistic Regression, Naive Bayes, and Random Forest
-> select Random Forest based on validation F1
-> save trained artifacts
-> load artifacts in Flask
-> receive user claim from React
-> transform input into same features
-> predict REAL or FAKE
-> return confidence and explanation
-> display result on frontend
```

The most important point to remember for viva:

```text
This project uses a small set of understandable features and classical ML algorithms.
Random Forest is deployed because it uses both text and metadata and gives the best F1 score.
The app predicts based on learned patterns, not live fact verification.
```
