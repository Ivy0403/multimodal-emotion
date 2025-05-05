import joblib
import numpy as np
import pandas as pd

# Load Model
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr.pkl", "rb"))

# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emotions emoji dictionary
emotions_emoji_dict = {
    "anger": "",
    "disgust": "",
    "fear": "",
    "happy": "",
    "joy": "",
    "neutral": "",
    "sad": "",
    "sadness": "",
    "shame": "",
    "surprise": ""
}

# Main function to run prediction
def main():
    # Example text for prediction
    raw_text = "I am feeling very happy today!"

    # Predict emotion
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)

    # Output results
    print("Original Text:", raw_text)
    print("Prediction:", prediction, emotions_emoji_dict[prediction])
    print("Confidence:", np.max(probability))

    # Display prediction probabilities
    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
    proba_df_clean = proba_df.T.reset_index()
    proba_df_clean.columns = ["emotions", "probability"]
    print("\nPrediction Probabilities:")
    print(proba_df_clean)

if __name__ == '__main__':
    main()
