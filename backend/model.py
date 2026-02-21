import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def get_engine():
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(db_url)

def load_training_data():
    engine = get_engine()
    query = """
        SELECT 
            p.game_id,
            p.period,
            p.home_score,
            p.away_score,
            p.score_differential,
            p.time_remaining,
            p.event_type,
            p.player_name,
            p.team,
            g.home_team,
            g.away_team
        FROM play_by_play p
        JOIN games g ON p.game_id = g.game_id
    """
    df = pd.read_sql(query, engine)
    return df

def engineer_features(df):
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0)
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0)
    df['score_diff'] = df['home_score'] - df['away_score']
    df['time_seconds'] = df['time_remaining'].apply(parse_time)
    df['total_time_remaining'] = ((4 - df['period'].clip(upper=4)) * 720) + df['time_seconds']
    
    df = df.sort_values(['game_id', 'period', 'time_remaining'])
    df['home_scoring_run'] = df.groupby('game_id')['home_score'].diff(10).fillna(0)
    df['away_scoring_run'] = df.groupby('game_id')['away_score'].diff(10).fillna(0)
    df['momentum'] = df['home_scoring_run'] - df['away_scoring_run']
    
    # Final score is the last recorded score per game
    final_scores = df.groupby('game_id').last()[['home_score', 'away_score']].reset_index()
    final_scores.columns = ['game_id', 'final_home', 'final_away']
    df = df.merge(final_scores, on='game_id')
    
    df['final_diff'] = df['final_home'] - df['final_away']
    df['comeback'] = ((df['score_diff'] < 0) & (df['final_diff'] > 0)).astype(int)
    
    return df

def parse_time(time_str):
    try:
        if 'PT' in str(time_str):
            time_str = str(time_str).replace('PT', '').replace('S', '')
            mins, secs = time_str.split('M')
            return float(mins) * 60 + float(secs)
        return 0
    except:
        return 0

def train_model():
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} rows")
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Only train on situations where team is losing by 1-20 points
    df = df[(df['score_diff'] < 0) & (df['score_diff'] >= -20)]
    
    features = ['score_diff', 'total_time_remaining', 'period', 'momentum']
    target = 'comeback'
    
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    print(f"Training on {len(X)} samples")
    print(f"Comeback rate: {y.mean():.2%}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('comeback_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to comeback_model.pkl")

if __name__ == "__main__":
    train_model()
