# Movie Review Sentiment Analyzer

This sentiment analyzer is designed to determine whether movie reviews express positive sentiments in favor of the movie or negative sentiments indicating dislike towards the movie. It utilizes a feed-forward sequential neural network built in Python with TensorFlow and Keras, employing tokenization and sequence preprocessing techniques.

## Usage

To utilize the movie review sentiment analyzer, follow these steps:

1. **Data Collection**: The movie reviews data can be sourced from the TensorFlow dataset provided as an example.

2. **Preprocessing**: Employ TensorFlow Keras preprocessing tools to tokenize and convert the movie reviews into sequences for analysis.

3. **Model Building**: Build a feed-forward sequential neural network using TensorFlow and Keras. This model will be trained on the preprocessed movie review data.

4. **Training**: Train the model on the labeled movie review dataset to enable it to learn the patterns associated with positive and negative sentiments.

5. **Prediction**: Input movie reviews into the trained model to predict whether the sentiment expressed is positive or negative towards the movie.

## Components

- **Data Source**: Utilizes movie reviews data from the TensorFlow dataset.
- **Preprocessing**: TensorFlow Keras preprocessing is employed for tokenization and sequence conversion of movie reviews.
- **Model**: A feed-forward sequential neu
- **Evaluation**: Assess the performance of the sentiment analyzer through evaluation metrics such as accuracy, precision, recall, and F1-score.

## Dependencies

- Python
- TensorFlow
- Keras
