# AI-Portfolio
My AI/ML Freelance Projects

Initial project: Iris dataset classifier with 97% accuracy using scikit-learn. Ready for client ML tasks!
Basic chatbot with rule-based, case-insensitive responses—ready for client apps!
House price predictor using linear regression with R^2: 1.00—ready for client real estate tasks!
Text-to-image category predictor using RandomForest and TF-IDF—interactive, multi-category, continuous input loop, 100% test accuracy, 98% cross-validation, ready for client image tasks!
# Sentiment Analysis Model with Orion AI

This project implements a sentiment analysis model using BERT, developed by Orion AI (built by xAI), trained on a custom dataset of 50 reviews to classify text as positive, negative, or neutral.

## Features
- Trained for 10 epochs with early stopping and cosine learning rate scheduling.
- Achieves an evaluation loss of 0.7509.
- Saved model available for inference.
- Logo inspired by the Orion constellation, reflecting intelligence and exploration.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python sentiment_analyzer.py`
3. Test with input: Type a review, "exit" to quit, or "save" to export.

## Results
- Eval Loss Trend: Dropped from 1.1686 to 0.7509 over 10 epochs (see `combined_training_log.txt`).
- Sample Prediction: "this product is amazing" -> positive

## Future Improvements
- Add more diverse training data.
- Fine-tune hyperparameters for better accuracy.

## Logo
![Orion AI Logo](link-to-logo-if-uploaded)