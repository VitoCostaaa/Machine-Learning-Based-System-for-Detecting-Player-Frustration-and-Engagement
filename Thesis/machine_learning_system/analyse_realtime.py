import joblib                      # Used to load trained machine learning models
import numpy as np                # Library for numerical operations
import pandas as pd               # Library for handling tabular data
import os                         # Provides file system operations
import warnings                   # Used to control warning messages


# Suppress warnings for a cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)


def live_prediction(player_name, features):
    
    # Define path to the trained model
    model_path = f'models/detector_{player_name}.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        return f"Error: No training model of {player_name} found."

    # Load trained model
    model = joblib.load(model_path)
    
    # Define feature names (must match training data)
    feature_names = [
        "duration", "score", "clicks", "avg_interval",
        "var_interval", "iti", "panic_rate", "death_altitude"
    ]
    
    # Convert input features into DataFrame format
    features_df = pd.DataFrame([features], columns=feature_names)

    # Predict player state
    prediction = model.predict(features_df)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(features_df)[0]
    
    # Determine highest confidence score
    confidence = np.max(probabilities)

    # Return prediction result and confidence
    return {
        "state": prediction,
        "confidence": f"{confidence:.2%}"
    }


if __name__ == "__main__":
    
    # Define player to analyze
    current_player = "Vito" 
    
    # Define column names for dataset
    column_names = [
        "player", "duration", "score", "clicks",
        "avg_int", "var_int", "iti", "panic", "alt", "label"
    ]
    
    # Load dataset
    df = pd.read_csv('data/player_data.csv', names=column_names)
    
    # Filter all rounds for selected player
    player_all_rounds = df[df['player'] == current_player]

    # Check if data exists
    if not player_all_rounds.empty:
        
        print(f"--- Start Test for all {len(player_all_rounds)} Rounds of {current_player} ---")
        
        # Iterate through all rounds
        for i in range(len(player_all_rounds)):
            
            # Extract feature values (excluding player and label)
            features = player_all_rounds.iloc[i, 1:9].values.tolist()
            
            # Call prediction function
            result = live_prediction(current_player, features)
            
            # Output results
            print(f"\nRound {i+1}: ")
            print(f"Result: {result}")
            
        print(f"\n--- Test for {current_player} completed ---")
    
    else:
        # Error message if no data found
        print(f"Error: No data found for {current_player} in the CSV.")