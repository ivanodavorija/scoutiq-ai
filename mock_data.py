

def get_player_positions():
    """
    Vraća listu rječnika koji predstavljaju detekcije igrača na terenu.
    """
    detections = [
        {
            "frame": 100,
            "timestamp_sec": 4.0,
            "player_id": 10,
            "team": "home",
            "position": (25.5, 40.0)
        },
        {
            "frame": 100,
            "timestamp_sec": 4.0,
            "player_id": 7,
            "team": "away",
            "position": (60.0, 35.2)
        },
        {
            "frame": 100,
            "timestamp_sec": 4.0,
            "player_id": 4,
            "team": "home",
            "position": (15.0, 20.8)
        },
        {
            "frame": 100,
            "timestamp_sec": 4.0,
            "player_id": 9,
            "team": "away",
            "position": (75.2, 50.1)
        }
    ]
    return detections


def filter_by_team(detections, team_name):
    """
    Filtrira listu detekcija i vraća samo igrače traženog tima.
    """
    filtered = []
    for detection in detections:
        if detection["team"] == team_name:
            filtered.append(detection)
    return filtered


# Primer korištenja (pokretanje datoteke):
if __name__ == "__main__":
    all_detections = get_player_positions()
    
    # Filtriranje samo "home" tima
    home_players = filter_by_team(all_detections, "home")
    print("Domaći igrači:", home_players)