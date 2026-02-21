from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder
import pandas as pd
import time
from database import get_connection

def fetch_recent_games(num_games=800):   
    print("Fetching recent NBA games...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable='2025-26',
        league_id_nullable='00',
        season_type_nullable='Regular Season'
    )
    games = gamefinder.get_data_frames()[0]
    game_ids = games['GAME_ID'].unique()[:num_games]
    return game_ids

def fetch_and_store_playbyplay(game_id, game_info):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO games (game_id, game_date, home_team, away_team)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            game_id,
            game_info.get('GAME_DATE'),
            game_info.get('TEAM_ABBREVIATION'),
            game_info.get('MATCHUP', '').split(' ')[-1]
        ))

        # Retry up to 3 times
        for attempt in range(3):
            try:
                pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60)
                df = pbp.get_data_frames()[0]
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"Retrying game {game_id} (attempt {attempt + 2})...")
                time.sleep(5)

        for _, row in df.iterrows():
            score = row.get('scoreHome') or '0'
            score_away = row.get('scoreAway') or '0'

            cur.execute("""
                INSERT INTO play_by_play 
                (game_id, period, time_remaining, home_score, away_score, score_differential, event_type, player_name, team)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                game_id,
                row.get('period'),
                row.get('clock'),
                score,
                score_away,
                row.get('scoremargin', 0),
                row.get('actionType'),
                row.get('playerNameI'),
                row.get('teamTricode')
            ))
        conn.commit()
        print(f"Stored play-by-play for game {game_id}")
    except Exception as e:
        print(f"Error processing game {game_id}: {e}")
    finally:
        cur.close()
        conn.close()

def run_pipeline():
    print("Fetching recent NBA games...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable='2025-26',
        league_id_nullable='00',
        season_type_nullable='Regular Season'
    )
    games_df = gamefinder.get_data_frames()[0]
    game_ids = games_df['GAME_ID'].unique()
    print(f"Found {len(game_ids)} unique games")

    for game_id in game_ids:
        game_info = games_df[games_df['GAME_ID'] == game_id].iloc[0]
        fetch_and_store_playbyplay(game_id, game_info)
        time.sleep(2)
    print("Pipeline complete!")


if __name__ == "__main__":
    run_pipeline()