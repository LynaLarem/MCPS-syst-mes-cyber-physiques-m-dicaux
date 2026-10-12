import numpy as np
from datetime import datetime
from collections import namedtuple

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import Scenario
from simglucose.simulation.scenario import Action as MealAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump


# ============================================================
# CONFIGURATION PATIENT ADULTE
# ============================================================

PATIENT_NAME = "adult#001"

REPAS_JOURNEE = [
    {"min_in_day": 60,  "carbs": 50, "label": "Petit-dej 50g"},
    {"min_in_day": 180, "carbs": 70, "label": "Dejeuner 70g"},
    {"min_in_day": 300, "carbs": 25, "label": "Gouter 25g"},
    {"min_in_day": 420, "carbs": 60, "label": "Diner 60g"},
    {"min_in_day": 540, "carbs": 20, "label": "Collation 20g"},
]

ICR = 12.0
ISF = 50.0
CGM_CIBLE = 120.0
BOLUS_MIN = 0.3
BOLUS_MAX = 15.0

BASAL = 0.050
PREBOLUS_STEPS = 8

PANNE_DUREE = 30
MARCHE_DUREE = 25
PANNE_START_STEP = 10

MINUTES_PAR_JOUR = 600
MAX_STEPS = 800   # 40 h avec sample time = 3 min

Q_PROFILE = np.array([[2.5, 0.0], [0.0, 0.25]])
R_PROFILE = np.array([[25.0]])

Action = namedtuple("Action", ["basal", "bolus"])


# ============================================================
# GRILLE OPTIMISÉE
# ============================================================

K_FRACTIONS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.75, 1.00,
    1.25, 1.50, 2.00
]

H_MULTIPLES = [
    1, 2, 3, 4, 5, 6, 7, 8,
    10, 12, 15, 18, 20
]

CUSUM_WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10]

CONFIRM_STEPS_LIST = [1, 2, 3, 4]

SLOPE_MIN_LIST = [0.0, 0.3, 0.5, 0.8, 1.0]

DELAI_MAX_ACCEPTABLE = 180.0


# ============================================================
# SCÉNARIO SIMGLUCOSE
# ============================================================

class AdultScenario(Scenario):
    def __init__(self, start_time, repas, sample_time):
        self.start_time = start_time
        self.sample_time = sample_time
        self.meal_carbs = {
            int(r["min"] / sample_time): r["carbs"]
            for r in repas
        }

    def get_action(self, t):
        elapsed = (t - self.start_time).total_seconds() / 60
        step = int(round(elapsed / self.sample_time))
        return MealAction(meal=self.meal_carbs.get(step, 0))

    def reset(self):
        pass


def build_repas(sample_time, max_steps):
    repas = []
    total_min = max_steps * sample_time
    nb_jours = int(total_min / MINUTES_PAR_JOUR) + 2

    for jour in range(nb_jours):
        base_min = jour * MINUTES_PAR_JOUR
        for r in REPAS_JOURNEE:
            repas.append({
                "min": base_min + r["min_in_day"],
                "carbs": r["carbs"],
                "label": r["label"],
            })

    return repas


def calculer_bolus(carbs, cgm_val):
    if cgm_val < 70:
        return 0.0

    bolus_repas = carbs / ICR
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80 else 0.0

    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)


def pump_is_failed(step):
    if step < PANNE_START_STEP:
        return False

    cycle_pos = (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE)
    return cycle_pos < PANNE_DUREE


def insulin_controller(cgm_val, bolus_plan):
    db = BASAL

    if cgm_val > 170:
        db += 0.008
    if cgm_val > 210:
        db += 0.015
    if cgm_val < 100:
        db -= 0.008
    if cgm_val < 85:
        db -= 0.015
    if cgm_val < 70:
        db = 0.0

    return max(0.0, db), bolus_plan


# ============================================================
# SIMULATION 40 H
# ============================================================

def run_simulation(with_fault=False):
    start_time = datetime.now()

    env_tmp = T1DSimEnv(
        patient=T1DPatient.withName(PATIENT_NAME),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=AdultScenario(start_time, [{"min": 60, "carbs": 50}], 3)
    )

    sample_time = env_tmp.sample_time
    del env_tmp

    repas = build_repas(sample_time, MAX_STEPS)

    env = T1DSimEnv(
        patient=T1DPatient.withName(PATIENT_NAME),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=AdultScenario(start_time, repas, sample_time)
    )

    bolus_steps_map = {}

    for r in repas:
        step_repas = int(r["min"] / sample_time)
        step_bolus = max(0, step_repas - PREBOLUS_STEPS)
        bolus_steps_map[step_bolus] = r["carbs"]

    bolus_steps_sorted = sorted(bolus_steps_map.keys())

    s = env.reset()
    obs = s.observation

    cgm_values = []
    pump_failed_labels = []
    missed_bolus_labels = []

    for step in range(MAX_STEPS):
        cgm_val = float(obs.CGM)

        carbs_now = bolus_steps_map.get(step, 0)
        bolus_plan = calculer_bolus(carbs_now, cgm_val) if carbs_now > 0 else 0.0

        failed = pump_is_failed(step) if with_fault else False
        missed = failed and (carbs_now > 0)

        if failed:
            action = Action(basal=0.0, bolus=0.0)
        else:
            basal, bolus = insulin_controller(cgm_val, bolus_plan)
            action = Action(basal=basal, bolus=bolus)

        cgm_values.append(cgm_val)
        pump_failed_labels.append(failed)
        missed_bolus_labels.append(missed)

        s = env.step(action)
        obs = s.observation

        if s.done:
            break

    return (
        np.array(cgm_values),
        np.array(pump_failed_labels),
        np.array(missed_bolus_labels),
        bolus_steps_sorted,
        sample_time
    )


# ============================================================
# KALMAN GELÉ : RÉSIDUS LIBRES
# ============================================================

N_DEGEL = 3


def kalman_frozen_residuals(cgm_values, sample_time):
    dt = sample_time

    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = Q_PROFILE
    R = R_PROFILE

    x = np.array([[cgm_values[0]], [0.0]])
    P = np.eye(2) * 10.0

    residuals = []
    frozen = False
    neg_slope_count = 0
    prev_cgm = cgm_values[0]

    for cgm in cgm_values:
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        p_libre = float((H @ x_pred)[0, 0])
        ecart = cgm - p_libre

        pente = cgm - prev_cgm

        if pente < 0:
            neg_slope_count += 1
        else:
            neg_slope_count = 0

        if frozen and neg_slope_count >= N_DEGEL:
            frozen = False

            z = np.array([[cgm]])
            residual = z - H @ x_pred

            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ residual
            P = (np.eye(2) - K @ H) @ P_pred

            residuals.append(0.0)
            prev_cgm = cgm
            continue

        if ecart > 0 and not frozen:
            frozen = True
            neg_slope_count = 0

        if frozen:
            residuals.append(ecart)
            x = x_pred
            P = P_pred
        else:
            z = np.array([[cgm]])
            residual = z - H @ x_pred

            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ residual
            P = (np.eye(2) - K @ H) @ P_pred

            residuals.append(float(residual[0, 0]))

        prev_cgm = cgm

    return np.array(residuals)


# ============================================================
# OUTILS
# ============================================================

def moving_average(signal, window):
    if window <= 1:
        return signal.copy()

    out = np.zeros_like(signal, dtype=float)

    for i in range(len(signal)):
        start = max(0, i - window + 1)
        out[i] = np.mean(signal[start:i + 1])

    return out


def compute_slope(cgm_values, sample_time, window=3):
    slope = np.zeros(len(cgm_values))

    for i in range(window, len(cgm_values)):
        slope[i] = (cgm_values[i] - cgm_values[i - window]) / (window * sample_time)

    return slope


def build_detection_windows(missed_bolus_labels, bolus_steps_sorted, n_steps):
    missed_steps = [
        s for s in bolus_steps_sorted
        if s < n_steps and missed_bolus_labels[s]
    ]

    windows = []

    for ms in missed_steps:
        next_bolus_steps = [s for s in bolus_steps_sorted if s > ms]

        if next_bolus_steps:
            end = next_bolus_steps[0] - 1
        else:
            end = n_steps - 1

        windows.append((ms, end))

    return windows


def apply_confirmation(raw_alarm, confirm_steps):
    confirmed = np.zeros_like(raw_alarm, dtype=bool)
    count = 0

    for i, val in enumerate(raw_alarm):
        if val:
            count += 1
        else:
            count = 0

        if count >= confirm_steps:
            confirmed[i] = True

    return confirmed


def cusum_positive_scores(residuals, k):
    score = 0.0
    scores = []

    for r in residuals:
        score = max(0.0, score + r - k)
        scores.append(score)

    return np.array(scores)


def compute_metrics(alarm_normal, alarm_fault, missed_bolus_labels, bolus_steps_sorted, sample_time):
    """
    Logique stricte inter-bolus :
      - 1 bolus prévu = 1 fenêtre [bolus_i ; bolus_{i+1}[
      - si bolus_i est manqué : alerte avant bolus_{i+1} => VP, sinon FN
      - si bolus_i est présent : alerte dans la fenêtre => FP
      - la dernière fenêtre incomplète est ignorée car il n'y a pas de prochain bolus.
    FP est compté sur le scénario normal et sur les fenêtres non manquées du scénario panne.
    """
    n = min(len(alarm_normal), len(alarm_fault), len(missed_bolus_labels))
    vp = fp = fn = 0
    delays = []

    for i in range(len(bolus_steps_sorted) - 1):
        bs = bolus_steps_sorted[i]
        be = bolus_steps_sorted[i + 1]
        if bs >= n:
            continue
        be = min(be, n)

        normal_alerted = bool(np.any(alarm_normal[bs:be]))
        fault_alerted = bool(np.any(alarm_fault[bs:be]))
        missed = bool(missed_bolus_labels[bs]) if bs < len(missed_bolus_labels) else False

        # Scénario normal : tous les bolus sont présents. Toute alerte = FP.
        if normal_alerted:
            fp += 1

        # Scénario avec panne : décision par fenêtre du bolus courant.
        if missed:
            if fault_alerted:
                vp += 1
                idx = np.where(alarm_fault[bs:be])[0]
                if len(idx) > 0:
                    delays.append(idx[0] * sample_time)
            else:
                fn += 1
        else:
            if fault_alerted:
                fp += 1

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = (2 * precision * rappel / (precision + rappel)
          if (precision + rappel) > 0 else 0.0)
    delay_mean = float(np.mean(delays)) if delays else float("inf")

    return {
        "VP": vp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Rappel": rappel,
        "F1": f1,
        "Delay": delay_mean,
    }


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate_config(
    res_normal,
    res_fault,
    cgm_normal,
    cgm_fault,
    missed_bolus_labels,
    bolus_steps_sorted,
    sample_time,
    cusum_window,
    k_frac,
    h_mult,
    confirm_steps,
    slope_min,
):
    res_normal_f = moving_average(res_normal, cusum_window)
    res_fault_f = moving_average(res_fault, cusum_window)

    pos_normal = res_normal_f[res_normal_f > 0]

    if len(pos_normal) > 2:
        sigma = np.std(pos_normal)
    else:
        sigma = np.std(res_normal_f)

    k = k_frac * sigma
    h = h_mult * sigma

    score_normal = cusum_positive_scores(res_normal_f, k)
    score_fault = cusum_positive_scores(res_fault_f, k)

    slope_normal = compute_slope(cgm_normal, sample_time)
    slope_fault = compute_slope(cgm_fault, sample_time)

    raw_normal = (score_normal > h) & (slope_normal >= slope_min)
    raw_fault = (score_fault > h) & (slope_fault >= slope_min)

    alarm_normal = apply_confirmation(raw_normal, confirm_steps)
    alarm_fault = apply_confirmation(raw_fault, confirm_steps)

    m = compute_metrics(
        alarm_normal=alarm_normal,
        alarm_fault=alarm_fault,
        missed_bolus_labels=missed_bolus_labels,
        bolus_steps_sorted=bolus_steps_sorted,
        sample_time=sample_time
    )

    return {
        "cusum_window": cusum_window,
        "k_frac": k_frac,
        "h_mult": h_mult,
        "k": k,
        "h": h,
        "sigma": sigma,
        "confirm_steps": confirm_steps,
        "slope_min": slope_min,
        **m
    }


def selection_score(r):
    return (
        6 * r["FN"]
        + 2 * r["FP"]
        + 0.03 * r["Delay"],
        r["FN"],
        r["FP"],
        r["Delay"]
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== Calibration Kalman + CUSUM adulte — v2 optimisée 40 h ===")

    cgm_normal, _, _, bolus_steps_sorted, sample_time = run_simulation(with_fault=False)

    cgm_fault, _, missed_bolus_labels, _, _ = run_simulation(with_fault=True)

    n_steps = min(len(cgm_normal), len(cgm_fault))

    cgm_normal = cgm_normal[:n_steps]
    cgm_fault = cgm_fault[:n_steps]
    missed_bolus_labels = missed_bolus_labels[:n_steps]

    print(f"Sample time   : {sample_time} min")
    print(f"Durée         : {n_steps * sample_time:.0f} min ({n_steps * sample_time / 60:.1f} h)")
    print(f"Bolus manqués : {int(np.sum(missed_bolus_labels))}")

    res_normal = kalman_frozen_residuals(cgm_normal, sample_time)
    res_fault = kalman_frozen_residuals(cgm_fault, sample_time)

    pos = res_normal[res_normal > 0]

    results = []

    total_configs = (
        len(CUSUM_WINDOWS)
        * len(K_FRACTIONS)
        * len(H_MULTIPLES)
        * len(CONFIRM_STEPS_LIST)
        * len(SLOPE_MIN_LIST)
    )

    print(f"\nConfigurations testées : {total_configs}")

    for cw in CUSUM_WINDOWS:
        for kf in K_FRACTIONS:
            for hm in H_MULTIPLES:
                for conf in CONFIRM_STEPS_LIST:
                    for slope_min in SLOPE_MIN_LIST:
                        r = evaluate_config(
                            res_normal=res_normal,
                            res_fault=res_fault,
                            cgm_normal=cgm_normal,
                            cgm_fault=cgm_fault,
                            missed_bolus_labels=missed_bolus_labels,
                            bolus_steps_sorted=bolus_steps_sorted,
                            sample_time=sample_time,
                            cusum_window=cw,
                            k_frac=kf,
                            h_mult=hm,
                            confirm_steps=conf,
                            slope_min=slope_min
                        )
                        results.append(r)

    valid = [
        r for r in results
        if np.isfinite(r["Delay"])
        and r["Delay"] <= DELAI_MAX_ACCEPTABLE
    ]

    if not valid:
        valid = [r for r in results if np.isfinite(r["Delay"])]

    top = sorted(valid, key=selection_score)[:20]
    # Les métriques sont calculées en interne uniquement pour sélectionner la meilleure configuration.
    # Affichage simplifié : pas de tableau VP/FP/FN, précision, rappel ou F1.

    best = top[0]

    print("\n=== Paramètres retenus automatiquement ===")
    print(f"Fenêtre résidus CUSUM : {best['cusum_window']} steps = {best['cusum_window'] * sample_time:.0f} min")
    print(f"k fraction            : {best['k_frac']} × sigma = {best['k']:.6f} mg/dL")
    print(f"h multiple            : {best['h_mult']} × sigma = {best['h']:.6f} mg/dL")
    print(f"Confirmation          : {best['confirm_steps']} steps = {best['confirm_steps'] * sample_time:.0f} min")
    print(f"Pente minimale        : {best['slope_min']:.2f} mg/dL/min")

    print("\n--- Lignes à copier dans le code online ---")
    print(f"CUSUM_RES_WINDOW      = {best['cusum_window']}")
    print(f"CUSUM_K               = {best['k']:.6f}")
    print(f"CUSUM_H               = {best['h']:.6f}")
    print(f"KALMAN_CONFIRM_STEPS  = {best['confirm_steps']}")
    print(f"KALMAN_SLOPE_MIN      = {best['slope_min']:.6f}")
    print("\n=== Métriques obtenues ===")
    print(f"VP / FP / FN          : {best['VP']} / {best['FP']} / {best['FN']}")
    print(f"Précision             : {100*best['Precision']:.1f}%")
    print(f"Rappel                : {100*best['Rappel']:.1f}%")
    print(f"F1-score              : {100*best['F1']:.1f}%")
    print(f"Délai moyen détection : {best['Delay']:.1f} min")


if __name__ == "__main__":
    main()
