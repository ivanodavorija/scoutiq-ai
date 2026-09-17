import cv2
from ultralytics import YOLO

# Učitavamo lagani pretrenirani model (automatski će se preuzeti pri prvom pokretanju)
model = YOLO("yolov8n.pt")

def detect_players(image_path: str, output_path: str = None) -> dict:
    """
    Učitava sliku, detektira ljude (igrače) pomoću YOLOv8 modela,
    i opcionalno sprema sliku s nacrtanim okvirima.
    """
    # Pokrećemo detekciju (conf=0.25 znači minimalno 25% sigurnosti)
    results = model(image_path, conf=0.25)[0]
    
    player_boxes = []
    image = cv2.imread(image_path)
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        # Klasa 0 u COCO datasetu je 'person'
        if cls_id == 0:
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            
            player_boxes.append({
                "confidence": round(confidence, 2),
                "box": [round(c, 1) for c in coords]
            })
            
            # Ako želimo spremiti vizualni rezultat, crtamo zeleni pravokutnik
            if output_path and image is not None:
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Player {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Spremamo sliku s nacrtanim detekcijama
    if output_path and image is not None:
        cv2.imwrite(output_path, image)
        
    return {
        "detected_players_count": len(player_boxes),
        "detections": player_boxes
    }