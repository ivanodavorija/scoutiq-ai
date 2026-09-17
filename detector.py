"""
detector.py  —  Detekcija nogometasa + klasifikacija timova (Man Utd vs Brighton)

Glavne izmjene u odnosu na staru verziju:
  1. NMS vise ne "guta" zaklonjene igrace  (team-aware NMS, visoki IoU prag)
  2. Tiled / SAHI inferencija -> mali i udaljeni igraci se detektiraju
  3. Occlusion-aware uzorkovanje dresa -> igrac iza ne nasljedjuje boju igraca ispred
  4. Filtriranje publike preko maske terena (nuzno jer snizavamo conf prag)
  5. Puno robusnija HSV klasifikacija + k-means fallback + ispravna detekcija suca
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
#  KONFIGURACIJA  — sve rucke na jednom mjestu
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # ---- model -----------------------------------------------------------
    model_path: str = "yolov8x.pt"   # 'm' je preslab za okluzije; 'x' ili 'yolo11x.pt'
    device: Optional[str] = None     # None = auto, ili "cuda:0" / "cpu"

    # ---- inferencija -----------------------------------------------------
    conf: float = 0.10               # nisko: zaklonjeni igraci imaju nisku pouzdanost
    imgsz_full: int = 1280           # slika 1102x615 -> default 640 unisti male igrace
    tta: bool = True                 # test-time augmentation (multi-scale + flip)
    use_tiles: bool = True           # SAHI-stil rezanje slike
    tile: int = 640
    tile_overlap: float = 0.35
    max_det: int = 300

    # ---- spajanje detekcija (custom NMS) ---------------------------------
    iou_hard: float = 0.88           # iznad ovoga = sigurno duplikat, brisi
    iou_same_team: float = 0.55      # isti tim + ovoliko preklapanja = duplikat
    contain_thr: float = 0.90        # mala kutija "progutana" velikom
    size_ratio_dup: float = 0.35     # koliko slicne dimenzije da bi bio duplikat

    # ---- filtriranje ------------------------------------------------------
    min_box_h: int = 18              # manje od ovoga je sum / publika
    max_box_h_frac: float = 0.55     # veca od 55% visine slike = greska
    min_aspect: float = 1.1          # covjek je visi nego siri (h/w)
    max_aspect: float = 6.0
    pitch_margin: int = 12           # koliko px iznad terena jos prihvacamo stopala

    # ---- klasifikacija dresa ---------------------------------------------
    torso_top: float = 0.10          # ROI trupa unutar boxa (postotak visine)
    torso_bottom: float = 0.48
    torso_left: float = 0.18
    torso_right: float = 0.82
    min_valid_px: int = 25           # ispod ovoga nema smisla klasificirati
    min_ratio: float = 0.07          # min. udio piksela boje u validnim pikselima
    dominance: float = 1.25          # koliko pobjednik mora nadjacati gubitnika
    dominance_strict: float = 1.70   # ostrije, kad odluku zakljucavamo kao pouzdanu
    min_exclusive_frac: float = 0.40 # ekskluzivni dio mora biti barem toliko trupa
    occluder_cont_max: float = 0.95  # iznad ovoga susjed je vjerojatno ja sam

    # ---- HSV rasponi ------------------------------------------------------
    grass: Tuple = ((28, 25, 20), (95, 255, 255))
    red_lo: Tuple = ((0, 90, 55), (7, 255, 255))
    red_hi: Tuple = ((170, 90, 55), (180, 255, 255))
    blue:   Tuple = ((94, 55, 45), (132, 255, 255))
    skin:   Tuple = ((6, 35, 95), (26, 175, 255))   # ruke/lice/noge — NE broji se
    white_s_max: int = 45            # gace, carape, crte na terenu
    white_v_min: int = 175
    dark_s_max: int = 75             # sudac
    dark_v_max: int = 92

    colors: dict = field(default_factory=lambda: {
        "Team Red":  (0, 0, 255),
        "Team Blue": (255, 0, 0),
        "Referee":   (0, 255, 255),
        "Unknown":   (160, 160, 160),
    })


CFG = Config()


# --------------------------------------------------------------------------- #
#  POMOCNE GEOMETRIJSKE FUNKCIJE
# --------------------------------------------------------------------------- #

def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _containment(small, big) -> float:
    """Koliki dio 'small' kutije lezi unutar 'big' kutije."""
    ix1, iy1 = max(small[0], big[0]), max(small[1], big[1])
    ix2, iy2 = min(small[2], big[2]), min(small[3], big[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    area = (small[2] - small[0]) * (small[3] - small[1])
    return (iw * ih) / float(area) if area else 0.0


def _similar_size(a, b, tol: float) -> bool:
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    if min(wa, wb, ha, hb) <= 0:
        return False
    return (abs(wa - wb) / max(wa, wb) < tol) and (abs(ha - hb) / max(ha, hb) < tol)


# --------------------------------------------------------------------------- #
#  1) MASKA TERENA  — izbacuje publiku/tribine
# --------------------------------------------------------------------------- #

def build_pitch_mask(image: np.ndarray) -> np.ndarray:
    """Vraca binarnu masku najveceg zelenog podrucja (igraliste)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(CFG.grass[0]), np.array(CFG.grass[1]))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones(image.shape[:2], np.uint8) * 255

    biggest = max(cnts, key=cv2.contourArea)
    pitch = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(pitch, [cv2.convexHull(biggest)], -1, 255, cv2.FILLED)
    # malo prosirimo da uhvatimo igrace na samom rubu (uz aut liniju)
    pitch = cv2.dilate(pitch, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    return pitch


def _on_pitch(box, pitch_mask) -> bool:
    """Igrac je na terenu ako su mu stopala (donji rub) unutar maske."""
    H, W = pitch_mask.shape[:2]
    cx = int((box[0] + box[2]) / 2)
    for dy in range(0, CFG.pitch_margin + 1, 3):
        fy = min(H - 1, max(0, int(box[3]) - dy))
        for dx in (0, -0.25, 0.25):
            fx = int(np.clip(cx + dx * (box[2] - box[0]), 0, W - 1))
            if pitch_mask[fy, fx] > 0:
                return True
    return False


# --------------------------------------------------------------------------- #
#  2) KLASIFIKACIJA DRESA  — otporna na okluziju
# --------------------------------------------------------------------------- #

def _color_masks(hsv: np.ndarray):
    red = cv2.inRange(hsv, np.array(CFG.red_lo[0]), np.array(CFG.red_lo[1])) | \
          cv2.inRange(hsv, np.array(CFG.red_hi[0]), np.array(CFG.red_hi[1]))
    blue = cv2.inRange(hsv, np.array(CFG.blue[0]), np.array(CFG.blue[1]))
    grass = cv2.inRange(hsv, np.array(CFG.grass[0]), np.array(CFG.grass[1]))
    skin = cv2.inRange(hsv, np.array(CFG.skin[0]), np.array(CFG.skin[1]))

    s, v = hsv[:, :, 1], hsv[:, :, 2]
    white = ((s < CFG.white_s_max) & (v > CFG.white_v_min)).astype(np.uint8) * 255
    dark = ((s < CFG.dark_s_max) & (v < CFG.dark_v_max)).astype(np.uint8) * 255

    # crvena ima prednost pred "skin" maskom kad je jako zasicena
    red_strong = (red > 0) & (s > 120)
    skin[red_strong] = 0
    return red, blue, grass, skin, white, dark


def _gauss_map(shape, box) -> np.ndarray:
    """2D Gaussova 'tezina pripadnosti' centrirana na kutiju."""
    H, W = shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sx = max(2.0, 0.34 * (x2 - x1))
    sy = max(2.0, 0.42 * (y2 - y1))
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    gx = np.exp(-((xs - cx) ** 2) / (2 * sx * sx))
    gy = np.exp(-((ys - cy) ** 2) / (2 * sy * sy))
    return np.outer(gy, gx)


def ownership_weight(shape, box, neighbours) -> np.ndarray:
    """
    Za svaki piksel: kolika je vjerojatnost da pripada BAS ovom igracu,
    a ne nekom od preklopljenih susjeda. Rjesava problem "dres susjeda
    upada u moju kutiju" bez da uopce trebamo znati boje.
    """
    own = _gauss_map(shape, box)
    if not neighbours:
        return np.ones(shape[:2], np.float32)
    total = own.copy()
    for nb in neighbours:
        total += _gauss_map(shape, nb)
    return (own / np.maximum(total, 1e-6)).astype(np.float32)


def fit_perspective(boxes, iters: int = 200, tol: float = 0.18):
    """
    U TV kadru visina igraca gotovo linearno raste s redom u kojem su mu
    stopala:  h ~= a * y2 + b.  Robusno (RANSAC) fitamo taj pravac kako bi
    mogli prepoznati KRNJE detekcije (samo trup, samo noge, samo glava),
    koje su glavni izvor duplikata.
    Vraca (a, b) ili None ako nema dovoljno podataka.
    """
    pts = [(b[3], b[3] - b[1]) for b in boxes if (b[3] - b[1]) > 4]
    if len(pts) < 4:
        return None
    pts = np.array(pts, np.float32)
    rng = np.random.default_rng(0)
    best, best_in = None, 0
    n = len(pts)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        (y1_, h1), (y2_, h2) = pts[i], pts[j]
        if abs(y2_ - y1_) < 1e-3:
            continue
        a = (h2 - h1) / (y2_ - y1_)
        b = h1 - a * y1_
        if not (-0.05 < a < 0.60):          # fizikalno smislen nagib
            continue
        pred = a * pts[:, 0] + b
        ok = np.abs(pts[:, 1] - pred) <= tol * np.maximum(pred, 1.0)
        if ok.sum() > best_in:
            best_in, best = int(ok.sum()), (a, b, ok)
    if best is None or best_in < max(3, int(0.35 * n)):
        return None
    a, b, ok = best
    # dorada least-squares na inlierima
    X, Y = pts[ok, 0], pts[ok, 1]
    if len(X) >= 2 and X.std() > 1e-3:
        a, b = np.polyfit(X, Y, 1)
    return float(a), float(b)


def predicted_height(model, y2: float) -> float:
    a, b = model
    return max(4.0, a * y2 + b)


def complete_box(model, box):
    """
    Krnju kutiju (odsjecene noge) produzujemo do visine koju perspektiva
    predvidja. Rjesenje h iz  h = a*(y1+h)+b  ->  h = (a*y1+b)/(1-a).
    """
    a, b = model
    x1, y1, x2, y2 = box
    if a >= 0.95:
        return box
    h_exp = (a * y1 + b) / (1.0 - a)
    h_cur = y2 - y1
    if h_exp <= 0 or h_cur >= 0.85 * h_exp:
        return box
    return [x1, y1, x2, y1 + h_exp]


def _kmeans_fallback(bgr_pixels: np.ndarray) -> str:
    """Kad su HSV brojanja neodlucna: nadji dominantnu boju i usporedi s referentnima."""
    if len(bgr_pixels) < 20:
        return "Unknown"
    data = np.float32(bgr_pixels)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    k = min(3, len(data))
    _, labels, centers = cv2.kmeans(data, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)

    best, best_score = "Unknown", 0.0
    for i in np.argsort(-counts):
        c = np.uint8([[centers[i]]])
        h, s, v = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[0][0].astype(int)
        weight = counts[i] / len(data)
        if s < 50:
            continue
        if (h <= 8 or h >= 170) and v > 50:
            score = weight
            lab = "Team Red"
        elif 94 <= h <= 132 and v > 45:
            score = weight
            lab = "Team Blue"
        else:
            continue
        if score > best_score:
            best, best_score = lab, score
    return best


def analyze_jersey(image: np.ndarray,
                   box,
                   occluder_mask: Optional[np.ndarray] = None,
                   weight_map: Optional[np.ndarray] = None,
                   strict: bool = False) -> dict:
    """
    Puna analiza trupa. Vraca rjecnik jer pozivatelju ne treba samo labela,
    nego i JACINA DOKAZA — po njoj kasnije rangiramo tko prvi smije zakljucati
    svoju boju u grupi preklopljenih igraca.

    occluder_mask : 255 = piksel pripada drugom tijelu -> ignoriramo ga
    weight_map    : 0..1 iz ownership_weight() -> meko dijeljenje spornih piksela
    strict        : odluka se zakljucava, pa prolazi samo cvrst dokaz
    """
    none = {"label": "Unknown", "color": CFG.colors["Unknown"],
            "score": 0.0, "evidence": 0.0, "n_valid": 0, "red": 0.0, "blue": 0.0}

    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h <= 4 or w <= 2:
        return none

    ty1, ty2 = y1 + int(h * CFG.torso_top), y1 + int(h * CFG.torso_bottom)
    tx1, tx2 = x1 + int(w * CFG.torso_left), x1 + int(w * CFG.torso_right)
    ty2, tx2 = max(ty2, ty1 + 2), max(tx2, tx1 + 2)

    roi = image[max(0, ty1):ty2, max(0, tx1):tx2]
    if roi.size == 0:
        return none

    # normalizacija osvjetljenja (reflektori rade nered u V kanalu)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (4, 4)).apply(lab[:, :, 0])
    roi_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)

    red_m, blue_m, grass_m, skin_m, white_m, dark_m = _color_masks(hsv)

    occ = np.zeros(hsv.shape[:2], np.uint8)
    if occluder_mask is not None:
        sub = occluder_mask[max(0, ty1):ty2, max(0, tx1):tx2]
        if sub.shape == hsv.shape[:2]:
            occ = sub

    base_invalid = (grass_m > 0) | (white_m > 0) | (skin_m > 0)
    valid = (~(base_invalid | (occ > 0))).astype(np.uint8) * 255
    n_valid = int(cv2.countNonZero(valid))
    n_free = int(np.count_nonzero(~base_invalid))

    if strict:
        if n_valid < CFG.min_valid_px:
            return none
        # Ako je od trupa ostao samo tanki rub kutije, dokaz nije reprezentativan:
        # bas je taj rub pun trave, susjeda i pozadine.
        if occ.any() and n_free > 0 and (n_valid / n_free) < CFG.min_exclusive_frac:
            return none
    elif n_valid < CFG.min_valid_px and occ.any():
        valid = (~base_invalid).astype(np.uint8) * 255      # ventil
        n_valid = int(cv2.countNonZero(valid))

    if n_valid < CFG.min_valid_px:
        return none

    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    px_valid = n_valid
    if weight_map is not None:
        wm = weight_map[max(0, ty1):ty2, max(0, tx1):tx2]
        if wm.shape == sat.shape:
            sat = sat * wm
            n_valid = max(1.0, float(wm[valid > 0].sum()))

    vmask = valid > 0
    red_score = float(sat[(red_m > 0) & vmask].sum())
    blue_score = float(sat[(blue_m > 0) & vmask].sum())
    dark_ratio = float(((dark_m > 0) & vmask).sum()) / max(1, px_valid)

    red_r, blue_r = red_score / n_valid, blue_score / n_valid
    total = max(red_score + blue_score, 1e-6)

    out = {"n_valid": px_valid, "red": red_r, "blue": blue_r,
           "evidence": abs(red_score - blue_score)}

    # --- SUDAC: tamno I nezasiceno, a nema ni crvene ni plave -------------
    if dark_ratio > 0.55 and red_r < 0.06 and blue_r < 0.06:
        out.update(label="Referee", color=CFG.colors["Referee"],
                   score=dark_ratio, evidence=dark_ratio * px_valid)
        return out

    dom = CFG.dominance_strict if strict else CFG.dominance
    if red_r >= CFG.min_ratio and red_score > blue_score * dom:
        out.update(label="Team Red", color=CFG.colors["Team Red"],
                   score=red_score / total)
        return out
    if blue_r >= CFG.min_ratio and blue_score > red_score * dom:
        out.update(label="Team Blue", color=CFG.colors["Team Blue"],
                   score=blue_score / total)
        return out

    if strict:                      # nista se ne nagadja kad zakljucavamo
        return none

    px = roi_norm[vmask]
    lab2 = _kmeans_fallback(px)
    if lab2 != "Unknown":
        out.update(label=lab2, color=CFG.colors[lab2], score=0.45)
        return out
    if dark_ratio > 0.45:
        out.update(label="Referee", color=CFG.colors["Referee"], score=dark_ratio)
        return out
    return none


def classify_jersey(image, box, occluder_mask=None, weight_map=None,
                    strict=False, debug=False):
    """Tanki omotac oko analyze_jersey — vraca (labela, boja, pouzdanost)."""
    r = analyze_jersey(image, box, occluder_mask, weight_map, strict)
    if debug:
        print(f"   valid={r['n_valid']} red={r['red']:.3f} blue={r['blue']:.3f} "
              f"evidence={r['evidence']:.1f} -> {r['label']}")
    return r["label"], r["color"], r["score"]


# --------------------------------------------------------------------------- #
#  3) DETEKTOR
# --------------------------------------------------------------------------- #

class TeamDetector:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._model = None

    # ---- lijeno ucitavanje modela (da se modul moze importati bez GPU-a) --
    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.cfg.model_path)
        return self._model

    # ------------------------------------------------------------------ #
    def _predict(self, img: np.ndarray, imgsz: int, tta: bool):
        res = self.model.predict(
            img,
            conf=self.cfg.conf,
            iou=0.92,              # NAMJERNO visoko: interni NMS ne smije brisati
            imgsz=imgsz,           # zaklonjene igrace — spajamo sami, kasnije
            classes=[0],           # samo 'person'
            augment=tta,
            max_det=self.cfg.max_det,
            device=self.cfg.device,
            verbose=False,
        )[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append([x1, y1, x2, y2, float(b.conf[0])])
        return out

    def _tiles(self, W: int, H: int):
        t, ov = self.cfg.tile, self.cfg.tile_overlap
        step = max(1, int(t * (1 - ov)))
        xs = list(range(0, max(W - t, 0) + 1, step))
        ys = list(range(0, max(H - t, 0) + 1, step))
        if not xs or xs[-1] + t < W:
            xs.append(max(W - t, 0))
        if not ys or ys[-1] + t < H:
            ys.append(max(H - t, 0))
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                yield x, y, min(x + t, W), min(y + t, H)

    def _raw_detections(self, image: np.ndarray):
        H, W = image.shape[:2]
        dets = self._predict(image, self.cfg.imgsz_full, self.cfg.tta)

        if self.cfg.use_tiles:
            for (tx1, ty1, tx2, ty2) in self._tiles(W, H):
                tile = image[ty1:ty2, tx1:tx2]
                if tile.size == 0:
                    continue
                for d in self._predict(tile, self.cfg.tile, False):
                    dets.append([d[0] + tx1, d[1] + ty1,
                                 d[2] + tx1, d[3] + ty1, d[4]])
        return dets

    # ------------------------------------------------------------------ #
    def _prefilter(self, dets, image, pitch_mask):
        H, W = image.shape[:2]
        keep = []
        for x1, y1, x2, y2, conf in dets:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh < self.cfg.min_box_h:
                continue
            if bh > H * self.cfg.max_box_h_frac:
                continue
            ar = bh / bw
            if not (self.cfg.min_aspect <= ar <= self.cfg.max_aspect):
                continue
            if not _on_pitch((x1, y1, x2, y2), pitch_mask):
                continue
            keep.append([x1, y1, x2, y2, conf])
        return keep

    # ------------------------------------------------------------------ #
    def _merge(self, dets, image, model=None):
        """
        Team-aware NMS s tri zastite:

        1. PERSPEKTIVA  — kutija cija visina bitno odstupa od ocekivane za taj
                          red slike je KRNJA detekcija (samo trup / samo noge),
                          a ne novi igrac. Ide van.
        2. CONTAINMENT  — ako je jedna kutija >=85% unutar druge, to je duplikat.
                          Ovo pravilo NADJACAVA signal dubine; upravo je obrnuti
                          redoslijed prije propustao krnje kutije kao "igraca iza".
        3. EKSKLUZIVNA BOJA — tek kad nijedna kutija ne sadrzi drugu, gledamo
                          boju u nepreklopljenom dijelu i dubinu stopala.
        """
        dets = sorted(dets, key=lambda d: -d[4])
        kept = []

        for d in dets:
            box = d[:4]

            # --- zastita 1: je li kutija uopce smislene visine? -------------
            if model is not None:
                pred = predicted_height(model, box[3])
                ratio = (box[3] - box[1]) / pred
                if ratio < 0.62 or ratio > 1.55:
                    continue

            lab, _, _ = classify_jersey(image, box)
            dup = False

            for k in kept:
                iou = _iou(box, k["box"])
                c_in = _containment(box, k["box"])      # koliko sam JA u njemu
                c_out = _containment(k["box"], box)     # koliko je ON u meni
                cont = max(c_in, c_out)

                if iou < self.cfg.iou_same_team and cont < self.cfg.contain_thr:
                    continue                             # nema konflikta

                if iou >= self.cfg.iou_hard:
                    dup = True
                    break

                # --- zastita 2: sadrzavanje --------------------------------
                # Krnja kutija (pola tijela) i stvarni igrac IZA mogu oboje biti
                # ~90% unutar prednje kutije, pa sadrzavanje samo po sebi ne
                # razlikuje to dvoje. Razlikuje ih VISINA: krnja kutija je
                # prekratka za svoj red u slici, igrac iza nije.
                if cont >= 0.85:
                    inner = box if c_in >= c_out else k["box"]
                    if model is None:
                        dup = True
                        break
                    r = (inner[3] - inner[1]) / predicted_height(model, inner[3])
                    if not (0.80 <= r <= 1.35):
                        dup = True          # krnja / napuhana -> duplikat
                        break
                    # puna visina: mozda je stvarno igrac iza. Odlucuje BOJA,
                    # a signal dubine se ovdje NE koristi (bio bi krug).
                    occ = self._occluder_mask(image, k["box"], k["label"])
                    lab_x, _, sc_x = classify_jersey(image, box, occluder_mask=occ)
                    if lab_x != "Unknown" and lab_x != k["label"] and sc_x >= 0.60:
                        continue
                    dup = True
                    break

                # --- zastita 3: tek sad smijemo traziti "drugog igraca" ------
                occ = self._occluder_mask(image, k["box"], k["label"])
                lab_x, _, sc_x = classify_jersey(image, box, occluder_mask=occ)
                distinct = (lab_x != "Unknown"
                            and lab_x != k["label"]
                            and sc_x >= 0.60)

                h_min = max(1.0, min(box[3] - box[1], k["box"][3] - k["box"][1]))
                if abs(box[3] - k["box"][3]) > 0.30 * h_min:
                    distinct = True

                if distinct:
                    continue

                if _similar_size(box, k["box"], self.cfg.size_ratio_dup):
                    dup = True
                    break

            if not dup:
                kept.append({"box": [float(v) for v in box],
                             "conf": d[4], "label": lab})
        return kept

    # ------------------------------------------------------------------ #
    @staticmethod
    def _occluder_mask(image: np.ndarray, box, label: str) -> np.ndarray:
        """
        Maska piksela koji SIGURNO pripadaju igracu ispred.
        Ako mu znamo boju dresa -> maskiramo bas te piksele (precizno).
        Inace -> uska elipsa u sredini boxa (konzervativno, da ne pojedemo susjeda).
        """
        m = np.zeros(image.shape[:2], np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return m

        if label in ("Team Red", "Team Blue"):
            sub = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            if label == "Team Red":
                col = cv2.inRange(sub, np.array(CFG.red_lo[0]), np.array(CFG.red_lo[1])) | \
                      cv2.inRange(sub, np.array(CFG.red_hi[0]), np.array(CFG.red_hi[1]))
            else:
                col = cv2.inRange(sub, np.array(CFG.blue[0]), np.array(CFG.blue[1]))
            col = cv2.dilate(col, np.ones((3, 3), np.uint8), iterations=1)
            m[y1:y2, x1:x2] = col
            return m

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax = max(2, int((x2 - x1) * 0.30))     # uzak, ne cijela sirina boxa
        ay = max(2, int((y2 - y1) * 0.45))
        cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, cv2.FILLED)
        return m

    @staticmethod
    def _box_mask(shape, box) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(m, (max(0, x1), max(0, y1)), (x2, y2), 255, cv2.FILLED)
        return m

    @staticmethod
    def _is_occluder(me, other) -> bool:
        """
        Susjed se smije koristiti kao zaklanjac SAMO ako je stvarno drugo
        tijelo. Ako jedna kutija uglavnom lezi unutar druge, vjerojatno je
        rijec o istom igracu (zaostali duplikat ili krnja detekcija) — a
        brisanje njegove boje tada brise MOJ vlastiti dres i prevrne labelu.
        Upravo se to dogadjalo igracu iza crvenog.
        """
        if _iou(me, other) <= 0.02:
            return False
        if max(_containment(me, other), _containment(other, me)) >= CFG.occluder_cont_max:
            return False
        return True

    @staticmethod
    def _refine_box(image, box, label, model):
        """
        YOLO cesto oko para igraca nacrta preširoku kutiju koja guta i susjeda.
        Nakon sto znamo boju dresa, stisnemo kutiju vodoravno na stvarni raspon
        TE boje. Samo suzavamo, nikad ne sirimo.
        """
        if label not in ("Team Red", "Team Blue") or model is None:
            return box
        x1, y1, x2, y2 = [int(v) for v in box]
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 10:
            return box

        sub = cv2.cvtColor(image[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2HSV)
        if sub.size == 0:
            return box
        if label == "Team Red":
            m = cv2.inRange(sub, np.array(CFG.red_lo[0]), np.array(CFG.red_lo[1])) | \
                cv2.inRange(sub, np.array(CFG.red_hi[0]), np.array(CFG.red_hi[1]))
        else:
            m = cv2.inRange(sub, np.array(CFG.blue[0]), np.array(CFG.blue[1]))

        band = m[int(h * 0.05):int(h * 0.58), :]
        if band.size == 0:
            return box
        cols = (band > 0).sum(axis=0).astype(np.float32)
        if cols.max() < 2:
            return box

        peak = int(np.argmax(cols))
        thr = 0.25 * cols.max()
        lo = peak
        while lo > 0 and cols[lo - 1] >= thr:
            lo -= 1
        hi = peak
        while hi < len(cols) - 1 and cols[hi + 1] >= thr:
            hi += 1

        exp_w = predicted_height(model, y2) / 2.6      # tipičan omjer sirina:visina
        pad = max(2, int(0.18 * exp_w))
        nx1 = max(x1, x1 + lo - pad)
        nx2 = min(x2, x1 + hi + 1 + pad)
        if (nx2 - nx1) < 0.55 * exp_w or (nx2 - nx1) < 5:
            return box
        return [float(nx1), float(y1), float(nx2), float(y2)]

    def _reclassify_with_occlusion(self, players, image):
        """
        Grupa preklopljenih igraca rjesava se POHLEPNO, po jacini dokaza.

        Prvo se izracuna koliko svaki igrac ima nedvosmislenog (nepreklopljenog)
        dresa. Onaj s najjacim dokazom zakljuca svoju boju; njegova se boja
        potom obrise svima s kojima se preklapa, pa se rangiranje ponavlja.
        Tako se znanje siri od najjasnijeg slucaja prema najzaklonjenijem,
        umjesto da svi pogadjaju odjednom (sto je prije zavrsavalo pat-pozicijom
        u kojoj je zaklonjeni igrac nasljedjivao boju onoga ispred).
        """
        n = len(players)
        if n == 0:
            return players

        boxes = [p["box"] for p in players]
        neigh = {i: [j for j in range(n) if j != i and self._is_occluder(boxes[i], boxes[j])]
                 for i in range(n)}
        wmaps = {i: ownership_weight(image.shape, boxes[i], [boxes[j] for j in neigh[i]])
                 for i in range(n)}

        for p in players:
            p.update(label="Unknown", color=CFG.colors["Unknown"], team_score=0.0)

        def build_mask(i, locked):
            """Susjedi: zakljucanima brisemo tocno njihovu boju, ostalima cijelu
            kutiju (jos ne znamo sto je njihovo, pa ne riskiramo)."""
            occ = np.zeros(image.shape[:2], np.uint8)
            for j in neigh[i]:
                if j in locked:
                    m = self._occluder_mask(image, boxes[j], players[j]["label"])
                else:
                    m = self._box_mask(image.shape, boxes[j])
                occ = cv2.bitwise_or(occ, m)
            return occ

        locked = set()
        while len(locked) < n:
            best = None
            for i in range(n):
                if i in locked:
                    continue
                occ = build_mask(i, locked)
                r = analyze_jersey(image, boxes[i],
                                   occluder_mask=occ if occ.any() else None,
                                   weight_map=wmaps[i], strict=True)
                if r["label"] == "Unknown":
                    continue
                if best is None or r["evidence"] > best[1]["evidence"]:
                    best = (i, r)
            if best is None:
                break
            i, r = best
            players[i].update(label=r["label"], color=r["color"],
                              team_score=round(float(r["score"]), 3))
            locked.add(i)

        # --- tko je ostao neodlucen: blaza pravila, uz boje zakljucanih -----
        for i in range(n):
            if i in locked:
                continue
            occ = np.zeros(image.shape[:2], np.uint8)
            for j in neigh[i]:
                if j in locked:
                    occ = cv2.bitwise_or(
                        occ, self._occluder_mask(image, boxes[j], players[j]["label"]))
            r = analyze_jersey(image, boxes[i],
                               occluder_mask=occ if occ.any() else None,
                               weight_map=wmaps[i])
            players[i].update(label=r["label"], color=r["color"],
                              team_score=round(float(r["score"]), 3))
        return players

    # ------------------------------------------------------------------ #
    @staticmethod
    def _colour_blobs(image, region):
        """Najvece souvisle mrlje crvenog i plavog dresa u zadanom podrucju."""
        x1, y1, x2, y2 = [int(v) for v in region]
        x1, y1 = max(0, x1), max(0, y1)
        sub = image[y1:y2, x1:x2]
        if sub.size == 0:
            return {}
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        cand = {
            "Team Red": cv2.inRange(hsv, np.array(CFG.red_lo[0]), np.array(CFG.red_lo[1])) |
                        cv2.inRange(hsv, np.array(CFG.red_hi[0]), np.array(CFG.red_hi[1])),
            "Team Blue": cv2.inRange(hsv, np.array(CFG.blue[0]), np.array(CFG.blue[1])),
        }
        min_area = max(10.0, 0.015 * sub.shape[0] * sub.shape[1])
        out = {}
        for lab, m in cand.items():
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            k, _, stats, cents = cv2.connectedComponentsWithStats(m, 8)
            if k <= 1:
                continue
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[best, cv2.CC_STAT_AREA] >= min_area:
                out[lab] = (x1 + float(cents[best][0]),
                            float(stats[best, cv2.CC_STAT_AREA]))
        return out

    def _resolve_same_label_conflicts(self, players, image):
        """
        Zadnja linija obrane.

        Ako se dvije kutije jako preklapaju i OBJE tvrde istu boju, a u njihovoj
        uniji postoje mrlje OBIJU boja, onda je barem jedna labela kriva: dva
        tijela ne mogu dijeliti isti dres na istom mjestu. Tada boje dodijelimo
        po polozaju — svaka kutija dobiva mrlju koja je njenom sredistu blize,
        uz uvjet da se mrlje ne dupliciraju.

        Ovo rjesava slucaj u kojem je YOLO nacrtao jednu preširoku kutiju
        centriranu izmedju dva igraca: u njoj doslovno ima vise tudje boje nego
        vlastite, pa je nijedno brojanje piksela unutar kutije ne moze spasiti.
        """
        n = len(players)
        if n < 2:
            return players

        # grupe medjusobno preklopljenih kutija
        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i in range(n):
            for j in range(i + 1, n):
                if _iou(players[i]["box"], players[j]["box"]) > 0.25:
                    parent[find(i)] = find(j)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for g in groups.values():
            if len(g) < 2:
                continue
            labs = {players[i]["label"] for i in g}
            if len(labs) > 1 or labs == {"Referee"} or labs == {"Unknown"}:
                continue                       # vec su razliciti — nista za rijesiti

            bs = [players[i]["box"] for i in g]
            region = [min(b[0] for b in bs), min(b[1] for b in bs),
                      max(b[2] for b in bs), max(b[3] for b in bs)]
            # gledamo samo pojas trupa, ne noge i travu
            hh = region[3] - region[1]
            region = [region[0], region[1] + 0.08 * hh, region[2], region[1] + 0.55 * hh]

            blobs = self._colour_blobs(image, region)
            if len(blobs) < 2:
                continue                       # nema sukoba, grupa je stvarno jednobojna

            items = sorted(blobs.items(), key=lambda kv: kv[1][0])   # po x
            gs = sorted(g, key=lambda i: (players[i]["box"][0] + players[i]["box"][2]) / 2)

            if len(gs) == len(items):          # 1-na-1, minimiziramo ukupni pomak
                import itertools
                best, best_cost = None, None
                for perm in itertools.permutations(range(len(items))):
                    cost = sum(abs((players[gs[k]]["box"][0] + players[gs[k]]["box"][2]) / 2
                                   - items[perm[k]][1][0]) for k in range(len(gs)))
                    if best_cost is None or cost < best_cost:
                        best, best_cost = perm, cost
                pairing = {gs[k]: items[best[k]][0] for k in range(len(gs))}
            else:                              # vise kutija nego mrlja -> najbliza
                pairing = {i: min(items, key=lambda kv: abs(
                    (players[i]["box"][0] + players[i]["box"][2]) / 2 - kv[1][0]))[0]
                    for i in gs}

            for i, lab in pairing.items():
                if players[i]["label"] != lab:
                    players[i].update(label=lab, color=CFG.colors[lab],
                                      team_score=0.5)
        return players

    # ------------------------------------------------------------------ #
    def detect(self, image_path: str, output_path: Optional[str] = None) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        pitch = build_pitch_mask(image)
        raw = self._raw_detections(image)
        raw = self._prefilter(raw, image, pitch)

        # perspektivni model fitamo na pouzdanijem podskupu (manje krnjih kutija)
        strong = [r[:4] for r in raw if r[4] >= 0.35]
        model = fit_perspective(strong if len(strong) >= 5 else [r[:4] for r in raw])

        players = self._merge(raw, image, model)
        if model:
            for p in players:
                p["box"] = complete_box(model, p["box"])

        players = self._reclassify_with_occlusion(players, image)

        if model:                       # stisni kutije pa jos jednom potvrdi boje
            for p in players:
                p["box"] = self._refine_box(image, p["box"], p["label"], model)
            players = self._reclassify_with_occlusion(players, image)

        players = self._resolve_same_label_conflicts(players, image)

        if output_path:
            self.draw(image.copy(), players, output_path)

        return {
            "total_detected": len(players),
            "breakdown": {
                "team_red":  sum(p["label"] == "Team Red" for p in players),
                "team_blue": sum(p["label"] == "Team Blue" for p in players),
                "referee":   sum(p["label"] == "Referee" for p in players),
                "unknown":   sum(p["label"] == "Unknown" for p in players),
            },
            "entities": [
                {"label": p["label"],
                 "box": [int(v) for v in p["box"]],
                 "conf": round(p["conf"], 3),
                 "team_score": p.get("team_score", 0.0)}
                for p in players
            ],
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def draw(image, players, output_path):
        # crtamo od najdaljeg prema najblizem da natpisi ne prekrivaju igrace
        H, W = image.shape[:2]
        for p in sorted(players, key=lambda q: q["box"][3]):
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            color = p.get("color", CFG.colors.get(p["label"], (160, 160, 160)))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

            txt = {"Team Red": "RED", "Team Blue": "BLU",
                   "Referee": "REF"}.get(p["label"], "?")
            fs = float(np.clip((x2 - x1) / 70.0, 0.30, 0.55))   # font prema sirini boxa
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ly = y1 - 2 if y1 - th - 4 > 0 else y2 + th + 4
            lx = min(x1, W - tw - 4)
            cv2.rectangle(image, (lx, ly - th - 3), (lx + tw + 4, ly + 1),
                          color, cv2.FILLED)
            cv2.putText(image, txt, (lx + 2, ly - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(output_path, image)


# --------------------------------------------------------------------------- #
#  Kompatibilnost sa starim pozivom
# --------------------------------------------------------------------------- #

_DETECTOR: Optional[TeamDetector] = None


def detect_and_classify_teams(image_path: str, output_path: str = None) -> dict:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = TeamDetector(CFG)
    return _DETECTOR.detect(image_path, output_path)


if __name__ == "__main__":
    import json
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "frame.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "output.png"
    print(json.dumps(detect_and_classify_teams(src, dst), indent=2))
