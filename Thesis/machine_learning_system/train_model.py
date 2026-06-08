
import pandas as pd                                                     # Library for handling structured data (DataFrames)
import numpy as np                                                      # Library for numerical computations
import matplotlib.pyplot as plt                                         # Library for data visualization
from sklearn.model_selection import train_test_split                    # Splits dataset into training and test sets
from sklearn.ensemble import RandomForestClassifier                     # Random Forest algorithm (ensemble of decision trees)
from sklearn.metrics import accuracy_score, classification_report       # Evaluation metrics for model performance
import joblib                                                           # Used to save and load trained models
import os                                                               # Provides operating system functionalities (e.g., file handling)



def train():  # Main function for the model training process
    
    # Check if dataset exists
    if not os.path.exists('data/player_data.csv'):  
        print("No data found!")  # Error message if no gameplay data is available
        return  # Stop execution if file is missing

    # Define column names manually (since CSV has no header)
    column_names = ["player", "duration", "score", "clicks", "avg_interval", 
                    "var_interval", "iti", "panic_rate", "death_altitude", "label"] 
    
    # Load dataset
    df = pd.read_csv('data/player_data.csv', names=column_names)
    
    # List of players to train models for
    players = ['Vito']

    # Ensure output directories exist
    if not os.path.exists('models'): os.makedirs('models')
    if not os.path.exists('plots'): os.makedirs('plots')

    # Loop over each player
    for player in players:
        print(f"\n--- Training for: {player} ---")
        
        # Filter dataset for current player
        player_df = df[df['player'] == player]                     

        # Check if there is enough data for training
        if len(player_df) < 11: 
            print(f"Skip {player}: Not enough data records ({len(player_df)}).")  # Minimum data requirement
            continue
        
        # Separate features (X) and labels (y)
        X = player_df.drop(['label', 'player'], axis=1) 
        y = player_df['label'] 

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Create model with 100 decision trees                                                        
    model = RandomForestClassifier(n_estimators=100) 
    
    # Train the model using training data
    model.fit(X_train, y_train)                             
    
    # --- Model evaluation ---
    
    # Predict labels for test data
    y_pred = model.predict(X_test)                                      
    
    # Output evaluation results
    print(f"\nResult for {player}:")                                
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")   # Percentage of correct predictions
    print(classification_report(y_test, y_pred))               # Precision, recall, F1-score

   
    
    # --- Feature importance analysis ---
    
    # Identify which features contribute most to predictions
    importances = model.feature_importances_
    indices = np.argsort(importances)
        
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance: {player}")
    plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'plots/feature_importance_{player}.png')
    plt.close()

        
    
    # Save trained model to file
    joblib.dump(model, f'models/detector_{player}.pkl')
    print(f"Model safed: models/detector_{player}.pkl")


# Entry point of the script
if __name__ == "__main__":  
    train()  # Execute training function