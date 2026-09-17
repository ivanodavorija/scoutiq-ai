import cv2
import os

def get_video_info(video_path: str) -> dict:
    """Učitava video i vraća osnovne metapodatke."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "Ne mogu otvoriti video datoteku."}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "duration_seconds": round(duration_seconds, 2)
    }

def extract_first_frame(video_path: str, output_path: str) -> bool:
    """Izdvaja prvi frame iz videa i sprema ga kao sliku."""
    cap = cv2.VideoCapture(video_path)
    
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_path, frame)
    
    cap.release()
    return success

def extract_frames_per_second(video_path: str, output_folder: str, frame_rate: int = 1) -> int:
    """
    Izdvaja frameove iz videa u definiranom intervalu (default: 1 frame po sekundi).
    Vraća ukupan broj spremljenih slika.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback ako video nema definiran FPS
        
    interval = int(fps / frame_rate) if int(fps / frame_rate) > 0 else 1
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    return saved_count