

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



def calculate_team_center(detections):
    """
    Računa težište momčadi (prosječnu X i Y poziciju) na temelju proslijeđenih detekcija.
    Vraća tuple (prosjecni_x, prosjecni_y).
    """
    # Zaštita od prazne liste radi izbjegavanja dijeljenja s nulom (ZeroDivisionError)
    if not detections:
        return (0.0, 0.0)

    total_x = 0.0
    total_y = 0.0
    count = len(detections)

    # Zbrajanje svih x i y koordinata
    for det in detections:
        total_x += det["position"][0]
        total_y += det["position"][1]

    # Izračun prosjeka
    avg_x = total_x / count
    avg_y = total_y / count

    return (avg_x, avg_y)


# Ispravljeni testni blok na dnu datoteke:
if __name__ == "__main__":
    all_detections = get_player_positions()

    # 1. Filtriranje samo "home" tima
    home_players = filter_by_team(all_detections, "home")
    print("Domaći igrači:", home_players)

    # 2. Izračun težišta za "home" tim
    home_center = calculate_team_center(home_players)
    print("Težište domaće ekipe (x, y):", home_center)