# calibration_lisa_fast_interbolus_strict.py
# Calibration LISA rapide — logique inter-bolus stricte
# Usage :
#   py calibration_lisa_fast_interbolus_strict.py --patient enfant
#   py calibration_lisa_fast_interbolus_strict.py --patient ado
#   py calibration_lisa_fast_interbolus_strict.py --patient adulte

import argparse
import time
from datetime import datetime
from collections import namedtuple

import numpy as np
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import Scenario
from simglucose.simulation.scenario import Action as MealAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump

Action = namedtuple("Action", ["basal", "bolus"])

# ============================================================
# CONFIGS PATIENTS
# ============================================================

PATIENTS = {
    "enfant": {
        "name": "child#001",
        "basal": 0.030,
        "icr": 7.0,
        "isf": 50.0,
        "bolus_min": 0.5,
        "bolus_max": 12.0,
        "panne_duree": 28,
        "marche_duree": 22,
        "repas": [
            {"min_in_day": 60,  "carbs": 30, "label": "Petit-dej 30g"},
            {"min_in_day": 180, "carbs": 35, "label": "Dejeuner 35g"},
            {"min_in_day": 270, "carbs": 10, "label": "Collation 10g"},
            {"min_in_day": 390, "carbs": 30, "label": "Diner 30g"},
            {"min_in_day": 480, "carbs": 15, "label": "Collation 15g"},
        ],
    },
    "ado": {
        "name": "adolescent#001",
        "basal": 0.040,
        "icr": 15.0,
        "isf": 60.0,
        "bolus_min": 0.3,
        "bolus_max": 10.0,
        "panne_duree": 32,
        "marche_duree": 20,
        "repas": [
            {"min_in_day": 60,  "carbs": 45, "label": "Petit-dej 45g"},
            {"min_in_day": 180, "carbs": 60, "label": "Dejeuner 60g"},
            {"min_in_day": 300, "carbs": 20, "label": "Gouter 20g"},
            {"min_in_day": 420, "carbs": 55, "label": "Diner 55g"},
            {"min_in_day": 540, "carbs": 15, "label": "Collation 15g"},
        ],
    },
    "adulte": {
        "name": "adult#001",
        "basal": 0.050,
        "icr": 12.0,
        "isf": 50.0,
        "bolus_min": 0.3,
        "bolus_max": 15.0,
        "panne_duree": 30,
        "marche_duree": 25,
        "repas": [
            {"min_in_day": 60,  "carbs": 50, "label": "Petit-dej 50g"},
            {"min_in_day": 180, "carbs": 70, "label": "Dejeuner 70g"},
            {"min_in_day": 300, "carbs": 25, "label": "Gouter 25g"},
            {"min_in_day": 420, "carbs": 60, "label": "Diner 60g"},
            {"min_in_day": 540, "carbs": 20, "label": "Collation 20g"},
        ],
    },
}

# ============================================================
# PARAMÈTRES COMMUNS
# ============================================================

CGM_CIBLE = 120.0
PREBOLUS_STEPS = 8
PANNE_START_STEP = 10
MINUTES_PAR_JOUR = 600
MAX_STEPS = 800        # 40 h avec sample time = 3 min
LISA_PIE_ALPHA = 0.85

# Grille de calibration. Tu peux réduire ces listes si tu veux encore plus rapide.
LW_LIST = [10, 15, 20, 25, 30, 40, 50]
SW_LIST = [3, 4, 5, 6, 8]
P_GFM_LIST = [70, 75, 80, 85, 88, 90, 92, 95]
P_IFM_LIST = [30, 40, 50, 60, 70, 75, 80, 85]
P_SLOPE_LIST = [70, 75, 80, 85, 88, 90]
SCORE_MIN_LIST = [2, 3]
CONFIRM_STEPS_LIST = [3, 5, 7, 10, 12]

# Sélection : d'abord FP faible, rappel acceptable, puis délai.
FA_NIVEAUX = [
    (0, "idéal      (FA = 0)"),
    (1, "très bon   (FA ≤ 1)"),
    (2, "bon        (FA ≤ 2)"),
    (3, "correct    (FA ≤ 3)"),
    (5, "acceptable (FA ≤ 5)"),
]
RAPPEL_MIN = 0.50

# Si True, FP = alertes dans simulation normale + alertes sur fenêtres sans bolus manqué dans simulation panne.
COUNT_FP_FAULT_NONMISSED = True


# ============================================================
# SCÉNARIO
# ============================================================

class PatientScenario(Scenario):
    def __init__(self, start_time, repas, sample_time):
        self.start_time = start_time
        self.sample_time = sample_time
        self.meal_carbs = {int(r["min"] / sample_time): r["carbs"] for r in repas}

    def get_action(self, t):
        elapsed = (t - self.start_time).total_seconds() / 60
        step = int(round(elapsed / self.sample_time))
        return MealAction(meal=self.meal_carbs.get(step, 0))

    def reset(self):
        pass


def build_repas(cfg, sample_time, max_steps):
    repas = []
    total_min = max_steps * sample_time
    nb_jours = int(total_min / MINUTES_PAR_JOUR) + 2
    for jour in range(nb_jours):
        base_min = jour * MINUTES_PAR_JOUR
        for r in cfg["repas"]:
            repas.append({
                "min": base_min + r["min_in_day"],
                "carbs": r["carbs"],
                "label": r["label"],
            })
    return repas


def calculer_bolus(cfg, carbs, cgm_val):
    if cgm_val < 70.0:
        return 0.0
    bolus_repas = carbs / cfg["icr"]
    bolus_correction = (cgm_val - CGM_CIBLE) / cfg["isf"] if cgm_val >= 80.0 else 0.0
    return round(max(cfg["bolus_min"], min(cfg["bolus_max"], bolus_repas + bolus_correction)), 2)


def pump_is_failed(cfg, step):
    if step < PANNE_START_STEP:
        return False
    cycle_pos = (step - PANNE_START_STEP) % (cfg["panne_duree"] + cfg["marche_duree"])
    return cycle_pos < cfg["panne_duree"]


def insulin_controller(cfg, cgm_val, bolus_plan):
    db = cfg["basal"]
    # Adaptation légère par profil via basal seulement, comme tes codes.
    if cfg["name"] == "child#001":
        if cgm_val > 170: db += 0.005
        if cgm_val > 210: db += 0.010
        if cgm_val < 100: db -= 0.006
        if cgm_val < 85:  db -= 0.010
    elif cfg["name"] == "adolescent#001":
        if cgm_val > 170: db += 0.007
        if cgm_val > 210: db += 0.012
        if cgm_val < 100: db -= 0.007
        if cgm_val < 85:  db -= 0.012
    else:
        if cgm_val > 170: db += 0.008
        if cgm_val > 210: db += 0.015
        if cgm_val < 100: db -= 0.008
        if cgm_val < 85:  db -= 0.015
    if cgm_val < 70:
        db = 0.0
    return max(0.0, db), bolus_plan


def run_simulation(cfg, with_fault=False):
    start_time = datetime.now()

    env_tmp = T1DSimEnv(
        patient=T1DPatient.withName(cfg["name"]),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=PatientScenario(start_time, [{"min": 60, "carbs": cfg["repas"][0]["carbs"]}], 3),
    )
    sample_time = env_tmp.sample_time
    del env_tmp

    repas = build_repas(cfg, sample_time, MAX_STEPS)

    env = T1DSimEnv(
        patient=T1DPatient.withName(cfg["name"]),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=PatientScenario(start_time, repas, sample_time),
    )

    bolus_steps_map = {}
    for r in repas:
        step_repas = int(r["min"] / sample_time)
        step_bolus = max(0, step_repas - PREBOLUS_STEPS)
        bolus_steps_map[step_bolus] = r["carbs"]
    bolus_steps_sorted = sorted([s for s in bolus_steps_map.keys() if s < MAX_STEPS])

    s = env.reset()
    obs = s.observation

    cgm_values = np.zeros(MAX_STEPS, dtype=float)
    insulin_values = np.zeros(MAX_STEPS, dtype=float)
    pump_failed_arr = np.zeros(MAX_STEPS, dtype=bool)
    bolus_missed_arr = np.zeros(MAX_STEPS, dtype=bool)

    n = 0
    for step in range(MAX_STEPS):
        cgm_val = float(obs.CGM)
        carbs_now = bolus_steps_map.get(step, 0)
        bolus_plan = calculer_bolus(cfg, carbs_now, cgm_val) if carbs_now > 0 else 0.0
        failed = pump_is_failed(cfg, step) if with_fault else False

        if failed:
            basal, bolus = 0.0, 0.0
        else:
            basal, bolus = insulin_controller(cfg, cgm_val, bolus_plan)

        cgm_values[step] = cgm_val
        insulin_values[step] = basal * sample_time + bolus
        pump_failed_arr[step] = failed
        bolus_missed_arr[step] = bool(failed and carbs_now > 0)

        s = env.step(Action(basal=basal, bolus=bolus))
        obs = s.observation
        n = step + 1
        if s.done:
            break

    return (
        cgm_values[:n],
        insulin_values[:n],
        pump_failed_arr[:n],
        bolus_missed_arr[:n],
        [s for s in bolus_steps_sorted if s < n],
        sample_time,
    )


# ============================================================
# CALCULS VECTORIZÉS
# ============================================================

def causal_moving_average(x, window):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)

    # i < window : moyenne depuis 0
    m = min(window, len(x))
    if m > 0:
        out[:m] = c[1:m + 1] / np.arange(1, m + 1)
    if len(x) > window:
        out[window:] = (c[window + 1:] - c[1:len(x) - window + 1]) / window
    return out


def compute_pie(insulin_values, alpha=LISA_PIE_ALPHA):
    # Récursion courte, peu coûteuse.
    pie = np.zeros(len(insulin_values), dtype=float)
    for i in range(1, len(insulin_values)):
        pie[i] = alpha * pie[i - 1] + insulin_values[i]
    return pie


def compute_lisa_metrics_fast(cgm_values, insulin_values, sample_time, lw, sw):
    pie = compute_pie(insulin_values)

    glucose_lw = causal_moving_average(cgm_values, lw)
    glucose_sw = causal_moving_average(cgm_values, sw)
    pie_lw = causal_moving_average(pie, lw)
    pie_sw = causal_moving_average(pie, sw)

    diff = glucose_sw - glucose_lw

    # GFM cumulatif avec reset si diff <= 0.
    gfm = np.zeros(len(cgm_values), dtype=float)
    pos = diff > 0
    vals = diff * sample_time
    for i in range(1, len(cgm_values)):
        gfm[i] = gfm[i - 1] + vals[i] if pos[i] else 0.0

    ifm = np.zeros(len(cgm_values), dtype=float)
    mask = pie_lw > 1e-6
    ifm[mask] = np.maximum(0.0, (pie_lw[mask] - pie_sw[mask]) / pie_lw[mask])

    slope = np.zeros(len(cgm_values), dtype=float)
    slope[1:] = np.diff(glucose_sw) / sample_time

    return gfm, ifm, slope


def confirmed_alarm_fast(raw_alarm, confirm_steps):
    raw = raw_alarm.astype(np.int16)
    if confirm_steps <= 1:
        return raw_alarm.copy()
    kernel = np.ones(confirm_steps, dtype=np.int16)
    # Somme des confirm_steps derniers points à chaque instant.
    count = np.convolve(raw, kernel, mode="full")[:len(raw)]
    return count >= confirm_steps


def compute_alarms_fast(gfm, ifm, slope, seuil_gfm, seuil_ifm, seuil_slope, score_min, confirm_steps):
    score = (2 * (gfm > seuil_gfm).astype(np.int8)
             + (ifm > seuil_ifm).astype(np.int8)
             + (slope > seuil_slope).astype(np.int8))
    raw = score >= score_min
    return confirmed_alarm_fast(raw, confirm_steps)


# ============================================================
# ÉVALUATION INTER-BOLUS STRICTE
# ============================================================

def build_interbolus_intervals(bolus_steps_sorted, n_steps):
    # On ignore la dernière fenêtre si elle n'a pas de prochain bolus.
    intervals = []
    for i in range(len(bolus_steps_sorted) - 1):
        start = bolus_steps_sorted[i]
        end = bolus_steps_sorted[i + 1]
        if 0 <= start < end <= n_steps:
            intervals.append((start, end))  # end exclu
    return intervals


def evaluate_windows_strict(alarms, bolus_missed_arr, bolus_steps_sorted):
    intervals = build_interbolus_intervals(bolus_steps_sorted, len(alarms))
    vp = fp = fn = 0
    delays = []

    for start, end in intervals:
        missed = bool(bolus_missed_arr[start])
        idx = np.flatnonzero(alarms[start:end])
        alerted = len(idx) > 0

        if missed:
            if alerted:
                vp += 1
                delays.append(int(idx[0]))
            else:
                fn += 1
        else:
            if alerted:
                fp += 1

    return vp, fp, fn, delays, len(intervals)


def evaluate_config_cached(metrics_n, metrics_f, bmissed_n, bmissed_f, bolus_steps, sample_time,
                           p_gfm, p_ifm, p_slope, score_min, confirm_steps):
    gfm_n, ifm_n, slope_n = metrics_n
    gfm_f, ifm_f, slope_f = metrics_f

    pos_slope_n = slope_n[slope_n > 0]
    seuil_gfm = np.percentile(gfm_n, p_gfm)
    seuil_ifm = np.percentile(ifm_n, p_ifm)
    seuil_slope = np.percentile(pos_slope_n, p_slope) if len(pos_slope_n) else 0.0

    alarms_n = compute_alarms_fast(gfm_n, ifm_n, slope_n, seuil_gfm, seuil_ifm, seuil_slope,
                                   score_min, confirm_steps)
    alarms_f = compute_alarms_fast(gfm_f, ifm_f, slope_f, seuil_gfm, seuil_ifm, seuil_slope,
                                   score_min, confirm_steps)

    _, fp_normal, _, _, nb_normal = evaluate_windows_strict(alarms_n, bmissed_n, bolus_steps)
    vp, fp_fault, fn, delays, nb_fault = evaluate_windows_strict(alarms_f, bmissed_f, bolus_steps)

    fp = fp_normal + (fp_fault if COUNT_FP_FAULT_NONMISSED else 0)
    rappel = vp / max(1, vp + fn)
    precision = vp / max(1, vp + fp)
    f1 = 2 * precision * rappel / max(1e-9, precision + rappel)
    delay_min = float(np.mean(delays) * sample_time) if delays else float("inf")

    return {
        "p_gfm": p_gfm,
        "p_ifm": p_ifm,
        "p_slope": p_slope,
        "score_min": score_min,
        "confirm_steps": confirm_steps,
        "seuil_gfm": float(seuil_gfm),
        "seuil_ifm": float(seuil_ifm),
        "seuil_slope": float(seuil_slope),
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "rappel": rappel,
        "precision": precision,
        "f1": f1,
        "delai_detection_min": delay_min,
        "panne_detectee": np.isfinite(delay_min),
        "nb_windows": nb_fault,
    }


def select_best(results):
    for fa_max, fa_label in FA_NIVEAUX:
        candidats = [r for r in results if r["panne_detectee"] and r["rappel"] >= RAPPEL_MIN and r["fp"] <= fa_max]
        if candidats:
            best = min(candidats, key=lambda x: (
                x["fp"],
                -x["rappel"],
                x["delai_detection_min"],
                -x["confirm_steps"],
                x["score_min"],
            ))
            return best, fa_label

    candidats = [r for r in results if r["panne_detectee"]]
    if candidats:
        best = min(candidats, key=lambda x: (-x["rappel"], x["fp"], x["delai_detection_min"]))
        return best, f"fallback — rappel={100*best['rappel']:.1f}%, FA={best['fp']}"

    return None, "aucune détection"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", choices=["enfant", "ado", "adulte"], default="enfant")
    args = parser.parse_args()

    cfg = PATIENTS[args.patient]
    t0 = time.time()

    print(f"\n=== Calibration offline LISA RAPIDE — {args.patient.upper()} — inter-bolus strict ===")
    print("Logique : bolus manqué = panne ; détection avant prochain bolus = VP")
    print("Méthode : NumPy + métriques pré-calculées par couple (LW, SW)")
    print("Simulation en cours...")

    cgm_n, ins_n, _, bmissed_n, bolus_steps, sample_time = run_simulation(cfg, with_fault=False)
    cgm_f, ins_f, _, bmissed_f, _, _ = run_simulation(cfg, with_fault=True)

    n = min(len(cgm_n), len(cgm_f))
    cgm_n, ins_n, bmissed_n = cgm_n[:n], ins_n[:n], bmissed_n[:n]
    cgm_f, ins_f, bmissed_f = cgm_f[:n], ins_f[:n], bmissed_f[:n]
    bolus_steps = [s for s in bolus_steps if s < n]

    nb_intervals = max(0, len(bolus_steps) - 1)
    nb_missed_intervals = sum(1 for s in bolus_steps[:-1] if bmissed_f[s])

    print(f"Sample time   : {sample_time} min")
    print(f"Durée         : {n * sample_time:.0f} min ({n * sample_time / 60:.1f} h)")
    print(f"Bolus steps   : {len(bolus_steps)}")
    print(f"Fenêtres utiles inter-bolus : {nb_intervals}")
    print(f"Bolus manqués évalués      : {nb_missed_intervals}")

    total_configs = 0
    all_results = []

    # Pré-calcul par paire LW/SW : gros gain de temps.
    pairs = [(lw, sw) for lw in LW_LIST for sw in SW_LIST if sw < lw]
    print(f"\nPré-calcul métriques : {len(pairs)} couples (LW, SW)")

    for pair_idx, (lw, sw) in enumerate(pairs, start=1):
        metrics_n = compute_lisa_metrics_fast(cgm_n, ins_n, sample_time, lw, sw)
        metrics_f = compute_lisa_metrics_fast(cgm_f, ins_f, sample_time, lw, sw)

        for p_gfm in P_GFM_LIST:
            for p_ifm in P_IFM_LIST:
                for p_slope in P_SLOPE_LIST:
                    for score_min in SCORE_MIN_LIST:
                        for confirm_steps in CONFIRM_STEPS_LIST:
                            r = evaluate_config_cached(
                                metrics_n, metrics_f,
                                bmissed_n, bmissed_f,
                                bolus_steps, sample_time,
                                p_gfm, p_ifm, p_slope,
                                score_min, confirm_steps,
                            )
                            r.update({
                                "lw_steps": lw,
                                "sw_steps": sw,
                                "lw_min": lw * sample_time,
                                "sw_min": sw * sample_time,
                            })
                            all_results.append(r)
                            total_configs += 1

        if pair_idx % 5 == 0 or pair_idx == len(pairs):
            print(f"  Couples traités : {pair_idx}/{len(pairs)}")

    best, qualite = select_best(all_results)
    elapsed = time.time() - t0

    print(f"\nConfigurations testées : {total_configs}")
    print(f"Temps d'exécution      : {elapsed:.1f} s")

    if best is None:
        print("\nAucune configuration n'a détecté une panne.")
        return

    top = sorted(
        [r for r in all_results if r["panne_detectee"]],
        key=lambda x: (x["fp"], -x["rappel"], x["delai_detection_min"], -x["f1"])
    )[:10]

    print("\n--- Top 10 configurations ---")
    print("LW SW | P_GFM P_IFM P_Pente | Score Conf | VP FP FN | Prec  Rec   F1   | Delay | Seuils")
    for r in top:
        print(
            f"{r['lw_steps']:>2} {r['sw_steps']:>2} | "
            f"{r['p_gfm']:>5} {r['p_ifm']:>5} {r['p_slope']:>7} | "
            f"{r['score_min']:>5} {r['confirm_steps']:>4} | "
            f"{r['vp']:>2} {r['fp']:>2} {r['fn']:>2} | "
            f"{100*r['precision']:>5.1f} {100*r['rappel']:>5.1f} {100*r['f1']:>5.1f} | "
            f"{r['delai_detection_min']:>5.1f} | "
            f"GFM={r['seuil_gfm']:.3f}, IFM={r['seuil_ifm']:.4f}, Slope={r['seuil_slope']:.4f}"
        )

    print("\n=== Paramètres retenus ===")
    print(f"Qualité          : {qualite}")
    print(f"LW               : {best['lw_steps']} steps = {best['lw_min']:.0f} min")
    print(f"SW               : {best['sw_steps']} steps = {best['sw_min']:.0f} min")
    print(f"P_GFM / P_IFM / P_Pente : {best['p_gfm']} / {best['p_ifm']} / {best['p_slope']}")
    print(f"Score minimum    : {best['score_min']} / 4 points")
    print(f"Confirmation     : {best['confirm_steps']} steps = {best['confirm_steps'] * sample_time:.0f} min")
    print(f"Seuil GFM        : {best['seuil_gfm']:.6f}")
    print(f"Seuil IFM        : {best['seuil_ifm']:.6f}")
    print(f"Seuil Pente      : {best['seuil_slope']:.6f} mg/dL/min")
    print(f"VP / FP / FN     : {best['vp']} / {best['fp']} / {best['fn']}")
    print(f"Rappel           : {100*best['rappel']:.1f}%")
    print(f"Précision        : {100*best['precision']:.1f}%")
    print(f"Score F1         : {100*best['f1']:.1f}%")
    print(f"Délai détection  : {best['delai_detection_min']:.1f} min")

    print("\n─── Lignes à copier dans le code online ──────────────────────────────────")
    print(f"LISA_LW                 = {best['lw_steps']}")
    print(f"LISA_SW                 = {best['sw_steps']}")
    print(f"LISA_GFM_THRESHOLD      = {best['seuil_gfm']:.6f}")
    print(f"LISA_IFM_THRESHOLD      = {best['seuil_ifm']:.6f}")
    print(f"LISA_SLOPE_THRESHOLD    = {best['seuil_slope']:.6f}")
    print(f"LISA_SCORE_MIN          = {best['score_min']}")
    print(f"LISA_CONFIRMATION_STEPS = {best['confirm_steps']}")


if __name__ == "__main__":
    main()
