from fastapi import FastAPI
from pydantic import BaseModel
from mock_data import get_player_positions, filter_by_team, calculate_team_center

app = FastAPI(title="ScoutIQ AI API")


# Model za validaciju dolaznih podataka
class AnalysisRequest(BaseModel):
    match_id: int
    team_to_analyze: str  # "home" ili "away"


@app.get("/")
def read_root():
    return {"message": "ScoutIQ AI Engine is running!"}


@app.get("/analyze")
def analyze_match():
    # 1. Dohvati sve detekcije
    detections = get_player_positions()

    # 2. Podijeli igrače prema ekipi
    home_players = filter_by_team(detections, "home")
    away_players = filter_by_team(detections, "away")

    # 3. Izračunaj težište obje ekipe
    home_center = calculate_team_center(home_players)
    away_center = calculate_team_center(away_players)

    # 4. Vrati rezultate
    return {
        "status": "success",
        "total_detections": len(detections),
        "analytics": {
            "home_team": {
                "player_count": len(home_players),
                "center_position": {
                    "x": home_center[0],
                    "y": home_center[1]
                }
            },
            "away_team": {
                "player_count": len(away_players),
                "center_position": {
                    "x": away_center[0],
                    "y": away_center[1]
                }
            }
        }
    }


@app.post("/analyze/custom")
def analyze_custom_team(request: AnalysisRequest):
    # 1. Dohvati detekcije
    detections = get_player_positions()

    # 2. Odaberi traženu ekipu
    selected_team_players = filter_by_team(
        detections,
        request.team_to_analyze
    )

    # 3. Izračunaj težište
    center = calculate_team_center(selected_team_players)

    # 4. Vrati rezultat
    return {
        "match_id": request.match_id,
        "analyzed_team": request.team_to_analyze,
        "player_count": len(selected_team_players),
        "center_position": {
            "x": center[0],
            "y": center[1]
        }
    }