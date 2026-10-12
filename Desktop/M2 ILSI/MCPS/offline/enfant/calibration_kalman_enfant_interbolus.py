# calibration_kalman_enfant_interbolus.py
# Calibration offline Kalman — Patient ENFANT — simulation 40 h
# Version inter-bolus stricte + affichage métriques

import numpy as np
from datetime import datetime
from collections import namedtuple

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import Scenario
from simglucose.simulation.scenario import Action as MealAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump


PATIENT_NAME = "child#001"

REPAS_JOURNEE = [
    {"min_in_day": 60, "carbs": 30, "label": "Petit-dej 30g"},
    {"min_in_day": 180, "carbs": 35, "label": "Dejeuner 35g"},
    {"min_in_day": 270, "carbs": 10, "label": "Collation 10g"},
    {"min_in_day": 390, "carbs": 30, "label": "Diner 30g"},
    {"min_in_day": 480, "carbs": 15, "label": "Collation 15g"},
]

ICR = 7.0
ISF = 50.0
CGM_CIBLE = 120.0
BOLUS_MIN = 0.5
BOLUS_MAX = 12.0

BASAL = 0.030
PREBOLUS_STEPS = 8
PANNE_DUREE = 28
MARCHE_DUREE = 22
PANNE_START_STEP = 10
MINUTES_PAR_JOUR = 600
MAX_STEPS = 800

Q_PROFILE = np.array([[1.5, 0.0], [0.0, 0.15]])
R_PROFILE = np.array([[25.0]])

FENETRES = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]
PERCENTILES = [85, 88, 90, 92, 94, 95, 96, 97, 98, 99]
CONFIRM_STEPS_LIST = [1, 2, 3, 4]
SLOPE_MIN_LIST = [0.0, 0.2, 0.5, 1.0]
SLOPE_LOOKBACK_STEPS = 3

W_FN = 3.0
W_FP = 1.0
W_DELAY = 0.02
DELAI_MAX_ACCEPTABLE = 180.0

Action = namedtuple("Action", ["basal", "bolus"])


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
            })

    return repas


def calculer_bolus(carbs, cgm_val):
    if cgm_val < 70.0:
        return 0.0

    bolus_repas = carbs / ICR
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0

    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)


class PatientScenario(Scenario):
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


def pump_is_failed(step):
    if step < PANNE_START_STEP:
        return False

    return (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE) < PANNE_DUREE


def insulin_controller(cgm_val, bolus_plan):
    db = BASAL

    if cgm_val > 170:
        db += 0.005
    if cgm_val > 210:
        db += 0.010
    if cgm_val < 100:
        db -= 0.006
    if cgm_val < 85:
        db -= 0.010
    if cgm_val < 70:
        db = 0.0

    return max(0.0, db), bolus_plan


def run_simulation(with_fault=False):
    start_time = datetime.now()

    env_tmp = T1DSimEnv(
        patient=T1DPatient.withName(PATIENT_NAME),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=PatientScenario(start_time, [{"min": 60, "carbs": 30}], 3),
    )

    sample_time = env_tmp.sample_time
    del env_tmp

    repas = build_repas(sample_time, MAX_STEPS)

    env = T1DSimEnv(
        patient=T1DPatient.withName(PATIENT_NAME),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=PatientScenario(start_time, repas, sample_time),
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
            db, tb = insulin_controller(cgm_val, bolus_plan)
            action = Action(basal=db, bolus=tb)

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
        sample_time,
    )


N_DEGEL = 3


def kalman_frozen_scores(cgm_values, sample_time, window_size):
    dt = sample_time

    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])

    x = np.array([[cgm_values[0]], [0.0]])
    P = np.eye(2) * 10.0

    ecarts_fenetre = []
    scores = []

    frozen = False
    neg_slope_count = 0
    prev_cgm = cgm_values[0]

    for cgm in cgm_values:
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q_PROFILE

        p_libre = float((H @ x_pred)[0, 0])
        ecart = cgm - p_libre

        if cgm - prev_cgm < 0:
            neg_slope_count += 1
        else:
            neg_slope_count = 0

        if frozen and neg_slope_count >= N_DEGEL:
            frozen = False
            ecarts_fenetre = []

            z = np.array([[cgm]])
            residual = z - H @ x_pred

            S = H @ P_pred @ H.T + R_PROFILE
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ residual
            P = (np.eye(2) - K @ H) @ P_pred

            scores.append(0.0)
            prev_cgm = cgm
            continue

        if ecart > 0 and not frozen:
            frozen = True
            neg_slope_count = 0

        if frozen:
            ecarts_fenetre.append(abs(ecart))

            if len(ecarts_fenetre) > window_size:
                ecarts_fenetre.pop(0)

            score = float(np.mean(ecarts_fenetre))
            x = x_pred
            P = P_pred

        else:
            ecarts_fenetre = []

            z = np.array([[cgm]])
            residual = z - H @ x_pred

            S = H @ P_pred @ H.T + R_PROFILE
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ residual
            P = (np.eye(2) - K @ H) @ P_pred

            score = 0.0

        scores.append(score)
        prev_cgm = cgm

    return np.array(scores)


def compute_slope_mask(cgm_values, sample_time, slope_min):
    mask = np.zeros(len(cgm_values), dtype=bool)

    for i in range(SLOPE_LOOKBACK_STEPS, len(cgm_values)):
        slope = (
            cgm_values[i] - cgm_values[i - SLOPE_LOOKBACK_STEPS]
        ) / (SLOPE_LOOKBACK_STEPS * sample_time)

        if slope_min <= 0:
            mask[i] = slope > 0.0
        else:
            mask[i] = slope >= slope_min

    return mask


def apply_confirmation(raw_alarm, confirm_steps):
    alarm = np.zeros(len(raw_alarm), dtype=bool)
    consec = 0

    for i, active in enumerate(raw_alarm):
        consec = consec + 1 if active else 0

        if consec >= confirm_steps:
            alarm[i] = True

    return alarm


def evaluate_windows(alarms, missed_bolus_labels, bolus_steps_sorted, n_steps):
    vp = 0
    fp = 0
    fn = 0

    for i, bs in enumerate(bolus_steps_sorted):
        if bs >= n_steps:
            continue

        be = bolus_steps_sorted[i + 1] if i + 1 < len(bolus_steps_sorted) else n_steps
        be = min(be, n_steps)

        missed = bool(missed_bolus_labels[bs]) if bs < len(missed_bolus_labels) else False
        detected = bool(np.any(alarms[bs:be]))

        if missed:
            if detected:
                vp += 1
            else:
                fn += 1
        else:
            if detected:
                fp += 1

    return vp, fp, fn


def compute_first_delay(alarms, missed_bolus_labels, bolus_steps_sorted, sample_time, n_steps):
    delays = []

    for i, bs in enumerate(bolus_steps_sorted):
        if bs >= n_steps:
            continue

        if not missed_bolus_labels[bs]:
            continue

        be = bolus_steps_sorted[i + 1] if i + 1 < len(bolus_steps_sorted) else n_steps
        be = min(be, n_steps)

        idx = np.where(alarms[bs:be])[0]

        if len(idx) > 0:
            delays.append(idx[0] * sample_time)

    return (
        float(np.mean(delays)) if delays else float("inf"),
        len(delays) > 0,
    )


def evaluate(
    scores_normal,
    scores_fault,
    cgm_normal,
    cgm_fault,
    missed_bolus,
    bolus_steps,
    sample_time,
    window_size,
    percentile,
    confirm_steps,
    slope_min,
):
    threshold = np.percentile(scores_normal, percentile)

    raw_normal = scores_normal > threshold
    raw_fault = scores_fault > threshold

    raw_normal = raw_normal & compute_slope_mask(cgm_normal, sample_time, slope_min)
    raw_fault = raw_fault & compute_slope_mask(cgm_fault, sample_time, slope_min)

    alarm_normal = apply_confirmation(raw_normal, confirm_steps)
    alarm_fault = apply_confirmation(raw_fault, confirm_steps)

    n_steps = len(scores_normal)

    missed_normal = np.zeros(n_steps, dtype=bool)
    _, fp_normal, _ = evaluate_windows(
        alarm_normal,
        missed_normal,
        bolus_steps,
        n_steps,
    )

    vp, fp_fault, fn = evaluate_windows(
        alarm_fault,
        missed_bolus,
        bolus_steps,
        n_steps,
    )

    fp = fp_normal + fp_fault

    precision = vp / (vp + fp) if (vp + fp) else 0.0
    recall = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    delay, detected = compute_first_delay(
        alarm_fault,
        missed_bolus,
        bolus_steps,
        sample_time,
        n_steps,
    )

    objective = (
        W_FN * fn
        + W_FP * fp
        + (W_DELAY * delay if detected and np.isfinite(delay) else 9999)
    )

    nb_windows = int(
        sum(
            1
            for bs in bolus_steps
            if bs < len(missed_bolus) and missed_bolus[bs]
        )
    )

    return {
        "window": window_size,
        "percentile": percentile,
        "threshold": threshold,
        "confirm": confirm_steps,
        "slope_min": slope_min,
        "VP": vp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "delay": delay,
        "detected": detected,
        "objective": objective,
        "nb_windows": nb_windows,
    }


def main():
    print("\n=== Calibration Kalman optimisée — ENFANT — 40 h ===")

    cgm_normal, _, _, _, sample_time = run_simulation(False)
    cgm_fault, _, missed_bolus, bolus_steps, _ = run_simulation(True)

    print(f"Sample time : {sample_time} min")
    print(f"Durée       : {MAX_STEPS * sample_time:.0f} min ({MAX_STEPS * sample_time / 60:.1f} h)")
    print(f"Bolus manqués : {int(np.sum(missed_bolus))}")

    all_results = []

    for w in FENETRES:
        scores_normal = kalman_frozen_scores(cgm_normal, sample_time, w)
        scores_fault = kalman_frozen_scores(cgm_fault, sample_time, w)

        for p in PERCENTILES:
            for c in CONFIRM_STEPS_LIST:
                for smin in SLOPE_MIN_LIST:
                    all_results.append(
                        evaluate(
                            scores_normal,
                            scores_fault,
                            cgm_normal,
                            cgm_fault,
                            missed_bolus,
                            bolus_steps,
                            sample_time,
                            w,
                            p,
                            c,
                            smin,
                        )
                    )

    valid = [
        r for r in all_results
        if r["detected"] and r["delay"] <= DELAI_MAX_ACCEPTABLE
    ]

    pool = valid if valid else [r for r in all_results if r["detected"]]

    best = min(
        pool,
        key=lambda r: (
            r["objective"],
            r["FN"],
            r["FP"],
            r["delay"],
        ),
    )

    print("\n=== Paramètres retenus automatiquement ===")
    print(f"Fenêtre Kalman        : {best['window']} steps = {best['window'] * sample_time:.0f} min")
    print(f"Percentile            : {best['percentile']}")
    print(f"Seuil Kalman          : {best['threshold']:.4f} mg/dL")
    print(f"Confirmation          : {best['confirm']} steps = {best['confirm'] * sample_time:.0f} min")
    print(f"Pente minimale        : {best['slope_min']:.2f} mg/dL/min")

    print("\n=== Métriques obtenues ===")
    print(f"VP / FP / FN          : {best['VP']} / {best['FP']} / {best['FN']}")
    print(f"Précision             : {100 * best['precision']:.1f}%")
    print(f"Rappel                : {100 * best['recall']:.1f}%")
    print(f"F1-score              : {100 * best['f1']:.1f}%")
    print(f"Délai moyen détection : {best['delay']:.1f} min")

    print("\n--- Lignes à copier dans le code online ---")
    print(f"KALMAN_WINDOW         = {best['window']}")
    print(f"KALMAN_THRESHOLD      = {best['threshold']:.6f}")
    print(f"KALMAN_CONFIRM_STEPS  = {best['confirm']}")
    print(f"KALMAN_SLOPE_MIN      = {best['slope_min']:.6f}")


if __name__ == "__main__":
    main()