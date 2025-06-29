import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import pickle
from flask import Flask, render_template, request, jsonify

# Define the MIANAReviewInference class directly in app.py
class MIANAReviewInference:
    def __init__(self, base_dir='weights'):
        """
        Initialize the inference model with pre-trained weights and preprocessing artifacts

        Args:
            base_dir (str): Directory containing saved model artifacts
        """
        self.base_dir = base_dir

        # Load tokenizer
        tokenizer_path = os.path.join(base_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Load label encoder
        label_encoder_path = os.path.join(base_dir, 'label_encoder.joblib')
        self.label_encoder = joblib.load(label_encoder_path)

        # Model configuration parameters
        self.max_len = 100  # Must match training configuration
        self.max_words = 5000  # Must match training configuration

        # Create and load model
        self.model = self._load_model()

    def _load_model(self):
        """
        Recreate and load pre-trained model weights

        Returns:
            Loaded Keras model with pre-trained weights
        """
        from tensorflow.keras.layers import (
            Input, Embedding, LSTM, Dense, Dropout,
            Layer, MultiHeadAttention, LayerNormalization,
            GlobalAveragePooling1D
        )
        from tensorflow.keras.models import Model

        class UserProductInteractionLayer(Layer):
            def __init__(self, units, num_heads=4, **kwargs):
                super(UserProductInteractionLayer, self).__init__(**kwargs)
                self.units = units
                self.num_heads = num_heads

            def build(self, input_shape):
                self.multi_head_attention = MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.units
                )
                self.layer_norm = LayerNormalization()
                super(UserProductInteractionLayer, self).build(input_shape)

            def call(self, inputs):
                text_embed, user_embed, product_embed = inputs
                attention_output = self.multi_head_attention(
                    query=text_embed,
                    value=user_embed,
                    key=product_embed
                )
                interaction_output = self.layer_norm(text_embed + attention_output)
                return interaction_output

        # Recreate model architecture
        text_input = Input(shape=(self.max_len,), name='text_input')

        embedding = Embedding(
            len(self.tokenizer.word_index) + 1,
            256,
            input_length=self.max_len
        )(text_input)

        lstm1 = LSTM(256, return_sequences=True, name='word_level_lstm1')(embedding)
        lstm2 = LSTM(128, return_sequences=True, name='word_level_lstm2')(lstm1)

        interaction_layer = UserProductInteractionLayer(
            units=128,
            num_heads=4
        )([lstm2, lstm2, lstm2])

        multi_head_attention = MultiHeadAttention(
            num_heads=4,
            key_dim=128
        )(query=interaction_layer, value=interaction_layer)

        pooled = GlobalAveragePooling1D()(multi_head_attention)
        dropout = Dropout(0.5)(pooled)

        output = Dense(len(self.label_encoder.classes_), activation='softmax', name='output')(dropout)

        model = Model(inputs=text_input, outputs=output)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Load pre-trained weights
        weights_path = os.path.join(self.base_dir, 'miana_model.weights.h5')
        model.load_weights(weights_path)

        return model

    def preprocess_text(self, text):
        """
        Preprocess input text for model prediction

        Args:
            text (str): Input review text

        Returns:
            Preprocessed and padded text sequence
        """
        # Tokenize and pad the text
        text_sequence = self.tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(
            text_sequence,
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )

        return text_padded

    def predict(self, review_text, top_k=3):
        """
        Predict label for the given review text

        Args:
            review_text (str): Input review text
            top_k (int): Number of top predictions to return

        Returns:
            List of top predictions with probabilities
        """
        # Preprocess text
        processed_text = self.preprocess_text(review_text)

        # Predict probabilities
        prediction = self.model.predict(processed_text)[0]

        # Get top-k predictions
        top_indices = prediction.argsort()[-top_k:][::-1]
        top_predictions = [
            {
                'label': self.label_encoder.classes_[idx],
                'probability': float(prediction[idx])
            }
            for idx in top_indices
        ]

        return top_predictions


# Initialize Flask app
app = Flask(__name__)

# Initialize the model
# Assuming the weights directory is in the same folder as app.py
model = MIANAReviewInference(base_dir='weights')

# Model performance metrics
model_metrics = {
    'training_accuracy': 0.9820,
    'validation_accuracy': 0.9402,
    'training_loss': 0.0483,
    'validation_loss': 0.2159,
    'epochs': 5,
    'year': 2025
}

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', metrics=model_metrics)

@app.route('/about')
def about():
    """Render the about page with model metrics"""
    return render_template('about.html', metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict the label for a given review"""
    if request.method == 'POST':
        # Get the review text from the request
        review_text = request.form.get('review', '')
        
        if not review_text:
            return jsonify({'error': 'No review text provided'}), 400
        
        try:
            # Get predictions from the model
            predictions = model.predict(review_text)
            return jsonify({
                'predictions': predictions,
                'review': review_text,
                'timestamp': '2025-03-27T13:05:45Z'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
