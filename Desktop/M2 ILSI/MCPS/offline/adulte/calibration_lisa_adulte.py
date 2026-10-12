# calibration_lisa_adulte_v4.py
# ─────────────────────────────────────────────────────────────────────────────
# Calibration offline LISA — Patient ADULTE — VERSION v4 (logique enfant)
#
# Architecture identique au code enfant v4 :
#   - Fonctions séparées et explicites (compute_pie, moving_average,
#     compute_lisa_metrics, compute_alarms, evaluate_windows, evaluate_config)
#   - Boucles Python transparentes (pas de raccourcis NumPy cachés)
#   - Évaluation par fenêtres inter-bolus [bolus_k → bolus_{k+1}[
#   - Score pondéré GFM×2 + IFM×1 + Pente×1 + filtre de confirmation
#   - Sélection : minimise FP, maximise rappel, minimise délai
#
# Seuls les paramètres physiologiques sont adaptés à l'adulte :
#   ICR=12, ISF=50, BASAL=0.050, repas adaptés (quantités adulte)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from datetime import datetime
from collections import namedtuple

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import Scenario
from simglucose.simulation.scenario import Action as MealAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump


# ─── Profil patient ───────────────────────────────────────────────────────────
PATIENT_NAME = "adult#001"

# ─── Repas journaliers (5 repas/jour — représentatifs d'un adulte diabétique) ─
REPAS_JOURNEE = [
    {"min_in_day":  60, "carbs": 50, "label": "Petit-dej 50g"},
    {"min_in_day": 180, "carbs": 70, "label": "Dejeuner  70g"},
    {"min_in_day": 300, "carbs": 25, "label": "Gouter    25g"},
    {"min_in_day": 420, "carbs": 60, "label": "Diner     60g"},
    {"min_in_day": 540, "carbs": 20, "label": "Collation 20g"},
]

# ─── Paramètres bolus dynamique ───────────────────────────────────────────────
ICR       = 12.0   # g glucides / U  (adulte : ratio plus agressif)
ISF       = 50.0   # mg/dL / U
CGM_CIBLE = 120.0
BOLUS_MIN = 0.3
BOLUS_MAX = 15.0

# ─── Paramètres simulation ────────────────────────────────────────────────────
BASAL            = 0.050
PREBOLUS_STEPS   = 8
PANNE_DUREE      = 30     # steps pompe HS
MARCHE_DUREE     = 25     # steps pompe OK
PANNE_START_STEP = 10
MINUTES_PAR_JOUR = 600    # minutes par cycle (10 h — adapté aux 5 repas)
MAX_STEPS        = 800    # ~40 h → plusieurs cycles repas + pannes

# ─── Grille de calibration (identique enfant) ─────────────────────────────────
LW_LIST = [10, 15, 20, 25, 30, 40, 50]
SW_LIST = [3, 4, 5, 6, 8]

P_GFM_LIST   = [70, 75, 80, 85, 88, 90, 92, 95]
P_IFM_LIST   = [30, 40, 50, 60, 70, 75, 80, 85]
P_SLOPE_LIST = [70, 75, 80, 85, 88, 90]

SCORE_MIN_LIST     = [2, 3]
CONFIRM_STEPS_LIST = [3, 5, 7, 10, 12]

# Critères de qualité par fenêtres inter-bolus
FA_NIVEAUX = [
    (0,  "idéal      (FA = 0)"),
    (1,  "très bon   (FA ≤ 1)"),
    (2,  "bon        (FA ≤ 2)"),
    (3,  "correct    (FA ≤ 3)"),
    (5,  "acceptable (FA ≤ 5)"),
]
RAPPEL_MIN = 0.50   # au moins 50 % des fenêtres avec bolus manqué détectées

Action = namedtuple("Action", ["basal", "bolus"])


# ─── Construction des repas multi-jours ───────────────────────────────────────

def build_repas(sample_time, max_steps):
    repas = []
    total_min = max_steps * sample_time
    nb_jours  = int(total_min / MINUTES_PAR_JOUR) + 2
    for jour in range(nb_jours):
        base_min = jour * MINUTES_PAR_JOUR
        for r in REPAS_JOURNEE:
            repas.append({
                "min"  : base_min + r["min_in_day"],
                "carbs": r["carbs"],
                "label": r["label"],
            })
    return repas


def calculer_bolus(carbs, cgm_val):
    if cgm_val < 70.0:
        return 0.0
    bolus_repas      = carbs / ICR
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0
    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)


# ─── Scénario patient ─────────────────────────────────────────────────────────

class PatientScenario(Scenario):
    def __init__(self, start_time, repas, sample_time):
        self.start_time  = start_time
        self.repas       = repas
        self.sample_time = sample_time
        self.meal_carbs  = {int(r["min"] / sample_time): r["carbs"] for r in repas}

    def get_action(self, t):
        elapsed = (t - self.start_time).total_seconds() / 60
        step    = int(round(elapsed / self.sample_time))
        return MealAction(meal=self.meal_carbs.get(step, 0))

    def reset(self):
        pass


# ─── Logique de panne ─────────────────────────────────────────────────────────

def pump_is_failed(step):
    if step < PANNE_START_STEP:
        return False
    return (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE) < PANNE_DUREE


def insulin_controller(cgm_val, bolus_plan):
    db = BASAL
    if cgm_val > 170: db += 0.008
    if cgm_val > 210: db += 0.015
    if cgm_val < 100: db -= 0.008
    if cgm_val <  85: db -= 0.015
    if cgm_val <  70: db  = 0.0
    return max(0.0, db), bolus_plan


# ─── Simulation ───────────────────────────────────────────────────────────────

def run_simulation(with_fault=False):
    start_time = datetime.now()

    env_tmp = T1DSimEnv(
        patient  = T1DPatient.withName(PATIENT_NAME),
        sensor   = CGMSensor.withName("Dexcom", seed=1),
        pump     = InsulinPump.withName("Insulet"),
        scenario = PatientScenario(start_time, [{"min": 60, "carbs": 50}], 3)
    )
    sample_time = env_tmp.sample_time
    del env_tmp

    repas = build_repas(sample_time, MAX_STEPS)

    env = T1DSimEnv(
        patient  = T1DPatient.withName(PATIENT_NAME),
        sensor   = CGMSensor.withName("Dexcom", seed=1),
        pump     = InsulinPump.withName("Insulet"),
        scenario = PatientScenario(start_time, repas, sample_time)
    )

    # Mapping step_bolus → carbs (pré-bolus décalé avant le repas)
    bolus_steps_map = {}
    for r in repas:
        step_repas = int(r["min"] / sample_time)
        step_bolus = max(0, step_repas - PREBOLUS_STEPS)
        bolus_steps_map[step_bolus] = r["carbs"]

    s   = env.reset()
    obs = s.observation

    cgm_values       = []
    insulin_values   = []
    pump_failed_arr  = []
    bolus_missed_arr = []   # True si pompe HS ET bolus attendu ce step
    bolus_steps_list = []   # steps où un bolus est normalement déclenché

    for step in range(MAX_STEPS):
        cgm_val    = float(obs.CGM)
        carbs_now  = bolus_steps_map.get(step, 0)
        bolus_plan = calculer_bolus(carbs_now, cgm_val) if carbs_now > 0 else 0.0
        failed     = pump_is_failed(step) if with_fault else False

        if carbs_now > 0:
            bolus_steps_list.append(step)

        if failed:
            basal, bolus = 0.0, 0.0
        else:
            basal, bolus = insulin_controller(cgm_val, bolus_plan)

        insulin_total = basal * sample_time + bolus

        cgm_values.append(cgm_val)
        insulin_values.append(insulin_total)
        pump_failed_arr.append(failed)
        # Panne réelle = pompe HS ET bolus attendu à ce step précis
        bolus_missed_arr.append(failed and carbs_now > 0)

        action = Action(basal=basal, bolus=bolus)
        s      = env.step(action)
        obs    = s.observation
        if s.done:
            break

    return (
        np.array(cgm_values),
        np.array(insulin_values),
        np.array(pump_failed_arr),
        np.array(bolus_missed_arr),
        sorted(bolus_steps_list),
        sample_time
    )


# ─── Métriques LISA ───────────────────────────────────────────────────────────

def compute_pie(insulin_values, alpha=0.85):
    pie = np.zeros(len(insulin_values))
    for i in range(1, len(insulin_values)):
        pie[i] = alpha * pie[i - 1] + insulin_values[i]
    return pie


def moving_average(x, window):
    values = np.zeros(len(x))
    for i in range(len(x)):
        start = max(0, i - window + 1)
        values[i] = np.mean(x[start:i + 1])
    return values


def compute_lisa_metrics(cgm_values, insulin_values, sample_time, lw_steps, sw_steps):
    n          = len(cgm_values)
    pie        = compute_pie(insulin_values)
    glucose_lw = moving_average(cgm_values, lw_steps)
    glucose_sw = moving_average(cgm_values, sw_steps)
    pie_lw     = moving_average(pie, lw_steps)
    pie_sw     = moving_average(pie, sw_steps)

    # GFM cumulatif (remis à 0 si SW ≤ LW)
    gfm = np.zeros(n)
    for k in range(1, n):
        diff = glucose_sw[k] - glucose_lw[k]
        gfm[k] = gfm[k - 1] + diff * sample_time if diff > 0 else 0.0

    # IFM : déficit d'insuline active normalisé
    ifm = np.zeros(n)
    for k in range(n):
        if pie_lw[k] > 1e-6:
            ifm[k] = max(0.0, (pie_lw[k] - pie_sw[k]) / pie_lw[k])

    # Pente du glucose lissé SW
    slope = np.zeros(n)
    slope[1:] = np.diff(glucose_sw) / sample_time

    return gfm, ifm, slope


# ─── Score pondéré + filtre de confirmation ───────────────────────────────────

def compute_alarms(gfm, ifm, slope,
                   seuil_gfm, seuil_ifm, seuil_slope,
                   score_min, confirm_steps):
    """
    Score pondéré :  GFM→+2, IFM→+1, Pente→+1
    Alerte brute si score >= score_min.
    Alerte confirmée si alerte brute pendant >= confirm_steps consécutifs.
    """
    score     = (2 * (gfm   > seuil_gfm  ).astype(int) +
                 1 * (ifm   > seuil_ifm  ).astype(int) +
                 1 * (slope > seuil_slope).astype(int))
    raw_alarm = score >= score_min

    confirmed = np.zeros(len(gfm), dtype=bool)
    consec    = 0
    for k in range(len(gfm)):
        consec = consec + 1 if raw_alarm[k] else 0
        confirmed[k] = consec >= confirm_steps

    return confirmed


# ─── Évaluation par fenêtres inter-bolus ─────────────────────────────────────

def evaluate_windows(alarms, bolus_missed_arr, bolus_steps_sorted):
    """
    Compte VP, FP, FN selon la logique stricte inter-bolus.

    Règle :
      - une fenêtre est définie par [bolus_i ; bolus_{i+1}[ ;
      - si le bolus_i est manqué, une alerte avant bolus_{i+1} => VP, sinon FN ;
      - si le bolus_i est présent, une alerte dans la fenêtre => FP ;
      - la dernière fenêtre sans prochain bolus est ignorée.
    """
    vp, fp, fn = 0, 0, 0
    n = len(alarms)

    for i in range(len(bolus_steps_sorted) - 1):
        bs = bolus_steps_sorted[i]
        be = bolus_steps_sorted[i + 1]
        if bs >= n:
            continue
        be = min(be, n)

        alerted = bool(np.any(alarms[bs:be]))
        missed = bool(bolus_missed_arr[bs]) if bs < len(bolus_missed_arr) else False

        if missed:
            if alerted:
                vp += 1
            else:
                fn += 1
        else:
            if alerted:
                fp += 1

    return vp, fp, fn


# ─── Évaluation d'une configuration ──────────────────────────────────────────

def evaluate_config(
    cgm_normal, insulin_normal, bolus_missed_normal, bolus_steps_sorted,
    cgm_fault,  insulin_fault,  bolus_missed_fault,
    sample_time, lw_steps, sw_steps,
    p_gfm, p_ifm, p_slope,
    score_min, confirm_steps
):
    gfm_n, ifm_n, slope_n = compute_lisa_metrics(
        cgm_normal, insulin_normal, sample_time, lw_steps, sw_steps)
    gfm_f, ifm_f, slope_f = compute_lisa_metrics(
        cgm_fault, insulin_fault, sample_time, lw_steps, sw_steps)

    # Seuils calibrés sur scénario NORMAL
    pos_slope_n = slope_n[slope_n > 0]
    seuil_gfm   = np.percentile(gfm_n, p_gfm)
    seuil_ifm   = np.percentile(ifm_n, p_ifm)
    seuil_slope = np.percentile(pos_slope_n, p_slope) if len(pos_slope_n) > 0 else 0.0

    alarms_normal = compute_alarms(
        gfm_n, ifm_n, slope_n, seuil_gfm, seuil_ifm, seuil_slope, score_min, confirm_steps)
    alarms_fault  = compute_alarms(
        gfm_f, ifm_f, slope_f, seuil_gfm, seuil_ifm, seuil_slope, score_min, confirm_steps)

    # Évaluation stricte par fenêtres inter-bolus
    # FP = alertes sur bolus présents, dans le scénario normal ET dans les fenêtres non manquées du scénario panne.
    _, fp_normal, _ = evaluate_windows(alarms_normal, bolus_missed_normal, bolus_steps_sorted)
    vp, fp_fault, fn = evaluate_windows(alarms_fault, bolus_missed_fault, bolus_steps_sorted)
    fp = fp_normal + fp_fault

    rappel    = vp / max(1, vp + fn)
    precision = vp / max(1, vp + fp)
    f1        = 2 * precision * rappel / max(1e-6, precision + rappel)

    # Délai de détection : premier step d'alerte dans la première fenêtre avec bolus manqué
    delay_min = float("inf")
    detected  = False
    n_steps   = len(alarms_fault)
    for i, bs in enumerate(bolus_steps_sorted):
        be = bolus_steps_sorted[i + 1] if i + 1 < len(bolus_steps_sorted) else n_steps
        bs = min(bs, n_steps - 1)
        be = min(be, n_steps)
        if bs < len(bolus_missed_fault) and bolus_missed_fault[bs]:
            alarm_idxs = np.where(alarms_fault[bs:be])[0]
            if len(alarm_idxs) > 0:
                delay_min = alarm_idxs[0] * sample_time
                detected  = True
                break

    return {
        "lw_steps": lw_steps, "sw_steps": sw_steps,
        "lw_min": lw_steps * sample_time, "sw_min": sw_steps * sample_time,
        "p_gfm": p_gfm, "p_ifm": p_ifm, "p_slope": p_slope,
        "score_min": score_min, "confirm_steps": confirm_steps,
        "seuil_gfm": seuil_gfm, "seuil_ifm": seuil_ifm, "seuil_slope": seuil_slope,
        "vp": vp, "fp": fp, "fn": fn,
        "rappel": rappel, "precision": precision, "f1": f1,
        "panne_detectee": detected,
        "delai_detection_min": delay_min,
    }


# ─── Sélection du meilleur résultat ──────────────────────────────────────────

def select_best(all_results):
    for fa_max, fa_label in FA_NIVEAUX:
        candidats = [
            r for r in all_results
            if r["panne_detectee"]
            and r["rappel"] >= RAPPEL_MIN
            and r["fp"] <= fa_max
        ]
        if candidats:
            best = min(candidats, key=lambda x: (
                x["fp"],
                -x["rappel"],
                x["delai_detection_min"],
                -x["confirm_steps"],
            ))
            return best, fa_label

    # Fallback : meilleur rappel même avec FA
    candidats = [r for r in all_results if r["panne_detectee"]]
    if candidats:
        best = min(candidats, key=lambda x: (-x["rappel"], x["fp"]))
        return best, f"fallback — rappel={best['rappel']*100:.0f}%, FA={best['fp']}"

    return None, "aucune détection"


# ─── Programme principal ──────────────────────────────────────────────────────

def main():
    print("\n=== Calibration offline LISA — Patient ADULTE — v4 ===")
    print("Évaluation : fenêtres inter-bolus (VP/FP/FN)")
    print("Logique    : score pondéré GFM×2 + IFM×1 + Pente×1 + confirmation")
    print("Repas      : multi-jours (5 repas/jour), bolus dynamique ICR/ISF")
    print()
    print("Simulation en cours...")

    (cgm_n, ins_n, _, bmissed_n, bolus_steps_sorted, sample_time
     ) = run_simulation(with_fault=False)
    (cgm_f, ins_f, _, bmissed_f, _, _
     ) = run_simulation(with_fault=True)

    print(f"Sample time   : {sample_time} min")
    print(f"Durée         : {len(cgm_n) * sample_time:.0f} min ({len(cgm_n) * sample_time / 60:.1f} h)")
    print(f"Bolus steps   : {len(bolus_steps_sorted)}")
    print(f"Bolus manqués : {int(np.sum(bmissed_f))} steps")
    print()

    all_results = []
    total_configs = (len(LW_LIST) * len(SW_LIST) * len(P_GFM_LIST) *
                     len(P_IFM_LIST) * len(P_SLOPE_LIST) *
                     len(SCORE_MIN_LIST) * len(CONFIRM_STEPS_LIST))
    print(f"Calibration : jusqu'à {total_configs} configurations à tester...")
    print()

    for lw in LW_LIST:
        for sw in SW_LIST:
            if sw >= lw:
                continue
            for p_gfm in P_GFM_LIST:
                for p_ifm in P_IFM_LIST:
                    for p_slope in P_SLOPE_LIST:
                        for score_min in SCORE_MIN_LIST:
                            for confirm_steps in CONFIRM_STEPS_LIST:
                                r = evaluate_config(
                                    cgm_n, ins_n, bmissed_n, bolus_steps_sorted,
                                    cgm_f, ins_f, bmissed_f,
                                    sample_time, lw, sw,
                                    p_gfm, p_ifm, p_slope,
                                    score_min, confirm_steps
                                )
                                all_results.append(r)

    best, qualite = select_best(all_results)

    print(f"Configurations testées : {len(all_results)}")
    print()
    print("=== Paramètres retenus ===")

    if best is None:
        print("Aucune configuration satisfaisante trouvée.")
        return

    print(f"Qualité          : {qualite}")
    print(f"LW               : {best['lw_steps']} steps = {best['lw_min']:.0f} min")
    print(f"SW               : {best['sw_steps']} steps = {best['sw_min']:.0f} min")
    print(f"P_GFM / P_IFM / P_Pente : {best['p_gfm']} / {best['p_ifm']} / {best['p_slope']}")
    print(f"Score minimum    : {best['score_min']} / 4 points")
    print(f"Confirmation     : {best['confirm_steps']} steps = {best['confirm_steps']*sample_time:.0f} min")
    print(f"Seuil GFM        : {best['seuil_gfm']:.4f}")
    print(f"Seuil IFM        : {best['seuil_ifm']:.4f}")
    print(f"Seuil Pente      : {best['seuil_slope']:.4f} mg/dL/min")
    print(f"VP / FP / FN     : {best['vp']} / {best['fp']} / {best['fn']}")
    print(f"Rappel           : {best['rappel']*100:.1f}%")
    print(f"Précision        : {best['precision']*100:.1f}%")
    print(f"Score F1         : {best['f1']*100:.1f}%")
    print(f"Délai détection  : {best['delai_detection_min']:.1f} min (dans la 1ère fenêtre)")
    print()
    print("─── Lignes à copier dans le code online ──────────────────────────────────")
    print(f"LISA_LW                 = {best['lw_steps']}")
    print(f"LISA_SW                 = {best['sw_steps']}")
    print(f"LISA_GFM_THRESHOLD      = {best['seuil_gfm']:.6f}")
    print(f"LISA_IFM_THRESHOLD      = {best['seuil_ifm']:.6f}")
    print(f"LISA_SLOPE_THRESHOLD    = {best['seuil_slope']:.6f}")
    print(f"LISA_SCORE_MIN          = {best['score_min']}")
    print(f"LISA_CONFIRMATION_STEPS = {best['confirm_steps']}")
    print()
    print("─── Logique de décision (remplace la logique OU dans l'online) ───────────")
    print("  score = 2*(GFM > LISA_GFM_THRESHOLD)")
    print("        + 1*(IFM > LISA_IFM_THRESHOLD)")
    print("        + 1*(slope > LISA_SLOPE_THRESHOLD)")
    print("  raw_alarm = (score >= LISA_SCORE_MIN)")
    print("  # Compteur consécutif :")
    print("  consec = consec + 1 if raw_alarm else 0")
    print("  alerte_lisa = (consec >= LISA_CONFIRMATION_STEPS)")


if __name__ == "__main__":
    main()
