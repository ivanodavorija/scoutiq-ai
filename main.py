import shutil
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from mock_data import get_player_positions, filter_by_team, calculate_team_center
from video_processor import get_video_info, extract_first_frame
from video_processor import get_video_info, extract_first_frame, extract_frames_per_second

app = FastAPI(title="ScoutIQ AI API")

# Mapa za spremanje videa
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AnalysisRequest(BaseModel):
    match_id: int
    team_to_analyze: str

@app.get("/")
def read_root():
    return {"message": "ScoutIQ AI Engine is running!"}

@app.get("/analyze")
def analyze_match():
    detections = get_player_positions()
    home_players = filter_by_team(detections, "home")
    away_players = filter_by_team(detections, "away")
    
    home_center = calculate_team_center(home_players)
    away_center = calculate_team_center(away_players)
    
    return {
        "status": "success",
        "total_detections": len(detections),
        "analytics": {
            "home_team": {
                "player_count": len(home_players),
                "center_position": {"x": home_center[0], "y": home_center[1]}
            },
            "away_team": {
                "player_count": len(away_players),
                "center_position": {"x": away_center[0], "y": away_center[1]}
            }
        }
    }

@app.post("/analyze/custom")
def analyze_custom_team(request: AnalysisRequest):
    detections = get_player_positions()
    selected_team_players = filter_by_team(detections, request.team_to_analyze)
    center = calculate_team_center(selected_team_players)
    
    return {
        "match_id": request.match_id,
        "analyzed_team": request.team_to_analyze,
        "player_count": len(selected_team_players),
        "center_position": {"x": center[0], "y": center[1]}
    }

# NOVI ENDPOINT KOJI NEDOSTAJE:
@app.post("/upload-video")
def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    info = get_video_info(file_location)
    
    # Putanja do mape za sekvencu frameova
    frames_dir = os.path.join(UPLOAD_DIR, "extracted_frames")
    saved_frames_count = extract_frames_per_second(file_location, frames_dir, frame_rate=1)
        
    return {
        "filename": file.filename,
        "status": "Video successfully processed!",
        "video_info": info,
        "extracted_frames_count": saved_frames_count,
        "frames_folder": frames_dir
    }