# fusionadulte_metriques_24h.py
# Évaluation adulte sur 24 h : Kalman-CUSUM + LISA + fusion passive avec métriques inter-bolus.
import time
import threading
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime, timedelta
from collections import namedtuple

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import Scenario
from simglucose.simulation.scenario import Action as MealAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump

# CONFIGURATION PATIENT ADULTE


PATIENT_NAME = "adult#001"

REPAS_JOURNEE = [
    {"min_in_day":  60, "carbs": 50, "label": "Petit-dej\n50g"},
    {"min_in_day": 180, "carbs": 70, "label": "Dejeuner\n70g"},
    {"min_in_day": 300, "carbs": 25, "label": "Gouter\n25g"},
    {"min_in_day": 420, "carbs": 60, "label": "Diner\n60g"},
    {"min_in_day": 540, "carbs": 20, "label": "Collation\n20g"},
]

ICR       = 12.0
ISF       = 50.0
CGM_CIBLE = 120.0
BOLUS_MIN = 0.3
BOLUS_MAX = 15.0


def calculer_bolus(carbs, cgm_val):
    if cgm_val < 70.0:
        return 0.0
    bolus_repas      = carbs / ICR
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0
    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)


MINUTES_PAR_JOUR = 600   # 5 repas / 10h
REPAS            = []


def build_repas_for_horizon(sample_time, horizon_steps=10000):
    repas     = []
    total_min = horizon_steps * sample_time
    nb_jours  = int(total_min / MINUTES_PAR_JOUR) + 2
    for jour in range(nb_jours):
        base_min = jour * MINUTES_PAR_JOUR
        for r in REPAS_JOURNEE:
            repas.append({
                "min"   : base_min + r["min_in_day"],
                "carbs" : r["carbs"],
                "bolus" : None,
                "label" : r["label"],
            })
    return repas


PANNE_DUREE      = 30   # adulte : 30 steps HS
MARCHE_DUREE     = 25   # adulte : 25 steps OK
PANNE_START_STEP = 10
BASAL            = 0.050  
PREBOLUS_STEPS   = 8
STEP_DELAY       = 0.0   # 0.0 = exécution rapide ; mets 1.5 pour suivre le graphe en temps réel
SIM_DURATION_HOURS = 24

Action = namedtuple("Action", ["basal", "bolus"])

# PARAMÈTRES LISA — ADULTE


LISA_LW                 = 15       
LISA_SW                 = 8        
LISA_GFM_THRESHOLD      = 755.739902
LISA_IFM_THRESHOLD      = 0.027636
LISA_SLOPE_THRESHOLD    = 2.128712
LISA_SCORE_MIN          = 3
LISA_PIE_ALPHA          = 0.85
LISA_CONFIRMATION_STEPS = 3     


# PARAMÈTRES KALMAN CUSUM — ADULTE


CUSUM_RES_WINDOW     = 3
CUSUM_K              = 97.396446
CUSUM_H              = 389.585782
KALMAN_CONFIRM_STEPS = 4
# KALMAN_SLOPE_MIN supprimé de la décision CUSUM.
# La pente CGM reste utilisée seulement pour le dégel interne du filtre.

Q_PROFILE = np.array([[2.5, 0.0],
                       [0.0, 0.25]])
R_PROFILE = np.array([[25.0]])


# PARAMÈTRES MACHINE À ÉTATS FUSION

SEQ_WINDOW_STEPS = 40   # fenêtre inter-bolus ~120 min



# NOTIFICATIONS NTFY

NTFY_SUSPICION_URL     = "https://ntfy.sh/pompe_suspicion_detecteur"
NTFY_SUSPICION_SEQ_URL = "https://ntfy.sh/pompe_suspicion_sequentielles"
NTFY_PANNE_URL         = "https://ntfy.sh/pompe_panne_confirme"

NTFY_COOLDOWN_STEPS = 10

_ntfy_last_suspicion     = -999
_ntfy_last_suspicion_seq = -999
_ntfy_last_panne         = -999



def _send_ntfy(url, title, message, priority="default", tags="rotating_light,warning"):
    def _do_send():
        try:
            data = message.encode("utf-8")
            req  = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Title",        title)
            req.add_header("Priority",     priority)
            req.add_header("Tags",         tags)
            req.add_header("Content-Type", "text/plain; charset=utf-8")
            with urllib.request.urlopen(req, timeout=8) as resp:
                print(f"[NTFY] HTTP {resp.status} -> {url.split('/')[-1]}")
        except Exception as e:
            print(f"[NTFY] Erreur envoi notification : {e}")
    threading.Thread(target=_do_send, daemon=False).start()


def notifier_suspicion(step):
    global _ntfy_last_suspicion

    if step - _ntfy_last_suspicion >= NTFY_COOLDOWN_STEPS:
        _ntfy_last_suspicion = step

        _send_ntfy(
            NTFY_SUSPICION_URL,
            "Suspicion pompe [ADULTE]",
            "Anomalie detectee par un seul detecteur - surveillance renforcee.",
            priority="high",
            tags="warning,mag"
        )

        print(f"[NTFY] Notification SUSPICION envoyee (step {step})")


def notifier_suspicion_sequentielle(step, ecart_steps, sample_time):
    global _ntfy_last_suspicion_seq

    if step - _ntfy_last_suspicion_seq >= NTFY_COOLDOWN_STEPS:
        _ntfy_last_suspicion_seq = step

        ecart_min = abs(ecart_steps) * sample_time

        _send_ntfy(
            NTFY_SUSPICION_SEQ_URL,
            "Suspicion elevee - alertes successives [ADULTE]",
            f"Les deux detecteurs ont alerte successivement (ecart : {ecart_min:.0f} min).",
            priority="high",
            tags="warning,rotating_light"
        )

        print(f"[NTFY] Notification SUSPICION SEQUENTIELLE envoyee (step {step})")


def notifier_panne(step):
    global _ntfy_last_panne

    if step - _ntfy_last_panne >= NTFY_COOLDOWN_STEPS:
        _ntfy_last_panne = step

        _send_ntfy(
            NTFY_PANNE_URL,
            "PANNE CONFIRMEE [ADULTE]",
            "Les deux detecteurs signalent une panne simultanee - verifiez la pompe.",
            priority="urgent",
            tags="rotating_light,sos"
        )

        print(f"[NTFY] Notification PANNE CONFIRMEE envoyee (step {step})")



# UTILITAIRES SIMULATION


def pump_is_failed(step):
    if step < PANNE_START_STEP:
        return False
    return (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE) < PANNE_DUREE


class AdultScenario(Scenario):
    def __init__(self, start_time, repas, sample_time):
        self.start_time  = start_time
        self.repas       = repas
        self.sample_time = sample_time
        self.meal_steps  = {int(r["min"] / sample_time): r["carbs"] for r in repas}

    def get_action(self, t):
        elapsed = (t - self.start_time).total_seconds() / 60
        step    = int(round(elapsed / self.sample_time))
        return MealAction(meal=self.meal_steps.get(step, 0))

    def reset(self):
        pass

    def extend_if_needed(self, current_step, sample_time, horizon=500):
        max_step = max(self.meal_steps.keys()) if self.meal_steps else 0
        if current_step + horizon > max_step:
            last_min      = max_step * sample_time
            nb_jours_supp = int((current_step + horizon) * sample_time / MINUTES_PAR_JOUR) + 2
            for jour in range(nb_jours_supp):
                base_min = last_min + (jour + 1) * MINUTES_PAR_JOUR
                for r in REPAS_JOURNEE:
                    new_min  = base_min + r["min_in_day"]
                    new_step = int(new_min / sample_time)
                    if new_step not in self.meal_steps:
                        self.meal_steps[new_step] = r["carbs"]
                        self.repas.append({
                            "min"  : new_min,
                            "carbs": r["carbs"],
                            "bolus": None,
                            "label": r["label"],
                            "step" : new_step,
                        })


# DÉTECTEUR LISA — adulte


class LISADetector:
    def __init__(self, sample_time):
        self.sample_time     = sample_time
        self.cgm_history     = []
        self.pie_history     = []
        self._pie_current    = 0.0
        self._gfm_current    = 0.0
        self._sw_prev        = None
        self._or_consecutive = 0
        self._confirmed      = False

    def update(self, cgm_val, insulin_dose):
        dt = self.sample_time

        # PIE
        self._pie_current = LISA_PIE_ALPHA * self._pie_current + insulin_dose
        self.pie_history.append(self._pie_current)
        self.cgm_history.append(cgm_val)

        if len(self.cgm_history) < LISA_LW + 1:
            self._or_consecutive = 0
            self._confirmed      = False
            return 0.0, 0.0, 0.0, False, False, 0

        # Fenêtres LW et SW
        cgm_arr    = np.array(self.cgm_history[-LISA_LW:])
        pie_arr    = np.array(self.pie_history[-LISA_LW:])
        glucose_lw = float(np.mean(cgm_arr))
        glucose_sw = float(np.mean(self.cgm_history[-LISA_SW:]))
        pie_lw     = float(np.mean(pie_arr))
        pie_sw     = float(np.mean(self.pie_history[-LISA_SW:]))

        # GFM cumulatif
        diff = glucose_sw - glucose_lw
        if diff > 0:
            self._gfm_current = self._gfm_current + diff * dt
        else:
            self._gfm_current = 0.0
        gfm = self._gfm_current

        # IFM
        ifm = max(0.0, (pie_lw - pie_sw) / pie_lw) if pie_lw > 1e-6 else 0.0

        # Pente
        slope         = (glucose_sw - self._sw_prev) / dt if self._sw_prev is not None else 0.0
        self._sw_prev = glucose_sw

        # Score pondéré
        lisa_score = (2 * int(gfm   > LISA_GFM_THRESHOLD) +
                      1 * int(ifm   > LISA_IFM_THRESHOLD) +
                      1 * int(slope > LISA_SLOPE_THRESHOLD))

        score_signal = lisa_score >= LISA_SCORE_MIN

        # Filtre de confirmation
        if score_signal:
            self._or_consecutive += 1
        else:
            self._or_consecutive = 0
            self._confirmed      = False

        if self._or_consecutive >= LISA_CONFIRMATION_STEPS:
            self._confirmed = True

        return gfm, ifm, slope, self._confirmed, score_signal, self._or_consecutive

    def reset(self):
        self._gfm_current    = 0.0
        self._or_consecutive = 0
        self._confirmed      = False


# DÉTECTEUR KALMAN CUSUM — adulte


class KalmanDetector:
    """
    Détecteur Kalman + CUSUM positif — version adulte alignée avec
    calibration_kalman_adulte_interbolus.py.

    Important :
    - Le filtre utilise les paramètres adultes :
        CUSUM_RES_WINDOW, CUSUM_K, CUSUM_H,
        KALMAN_CONFIRM_STEPS.
    - Le gel Kalman sert à produire des résidus libres, comme dans la calibration.
      Il est activé dès qu'un écart positif apparaît (CGM > prédiction libre).
    - Le CUSUM positif est ensuite calculé sur la moyenne des résidus positifs.
    - L'alerte Kalman est confirmée si :
          S_k > CUSUM_H
          S_k > CUSUM_H
          pendant KALMAN_CONFIRM_STEPS steps consécutifs.
    - Le dégel est interne : si le CGM descend pendant N_DEGEL steps consécutifs,
      Kalman se remet en mode normal et suit à nouveau la mesure CGM.
    - La fusion reste passive : elle ne commande jamais Kalman.
    """

    N_DEGEL = 3
    SLOPE_LOOKBACK_STEPS = 3

    def __init__(self, first_cgm, sample_time):
        self.dt = sample_time
        self.x  = np.array([[float(first_cgm)], [0.0]])
        self.P  = np.eye(2) * 10.0
        self.F  = np.array([[1, self.dt], [0, 1]])
        self.H  = np.array([[1, 0]])
        self.Q  = Q_PROFILE
        self.R  = R_PROFILE

        self.freeze = False
        self._residual_window = []
        self._cusum_score = 0.0
        self._confirm_count = 0
        self._confirmed = False
        self._neg_slope_count = 0
        self._cgm_history = [float(first_cgm)]
        self._prev_cgm = float(first_cgm)

    def _slope_cgm(self):
        if len(self._cgm_history) <= self.SLOPE_LOOKBACK_STEPS:
            return 0.0
        return (
            self._cgm_history[-1] - self._cgm_history[-1 - self.SLOPE_LOOKBACK_STEPS]
        ) / (self.SLOPE_LOOKBACK_STEPS * self.dt)

    def _kalman_correction(self, x_pred, P_pred, z):
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        x_new = x_pred + K @ (z - self.H @ x_pred)
        P_new = (np.eye(2) - K @ self.H) @ P_pred
        return x_new, P_new

    def update(self, cgm):
        cgm = float(cgm)
        self._cgm_history.append(cgm)

        z = np.array([[cgm]])

        # 1) Prédiction libre
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        p_libre = float((self.H @ x_pred)[0, 0])
        ecart_libre = float(cgm - p_libre)
        residual_pos = max(0.0, ecart_libre)

        # 2) Détection de descente CGM pour le dégel
        delta_cgm = cgm - self._prev_cgm
        if delta_cgm < 0:
            self._neg_slope_count += 1
        else:
            self._neg_slope_count = 0

        just_degel = False
        if self.freeze and self._neg_slope_count >= self.N_DEGEL:
            # Dégel : Kalman recommence à suivre la mesure CGM.
            self.freeze = False
            self._residual_window = []
            self._cusum_score = 0.0
            self._confirm_count = 0
            self._confirmed = False
            self._neg_slope_count = 0
            just_degel = True

            self.x, self.P = self._kalman_correction(x_pred, P_pred, z)
            self._prev_cgm = cgm
            return p_libre, ecart_libre, 0.0, False

        # 3) Gel Kalman aligné avec la calibration :
        #    dès que CGM > prédiction libre, on passe en prédiction libre.
        if (ecart_libre > 0.0) and (not self.freeze):
            self.freeze = True
            self._neg_slope_count = 0

        # 4) Mise à jour CUSUM sur les résidus libres positifs
        if self.freeze:
            self._residual_window.append(residual_pos)
            if len(self._residual_window) > CUSUM_RES_WINDOW:
                self._residual_window.pop(0)

            residual_filtre = float(np.mean(self._residual_window)) if self._residual_window else 0.0
            self._cusum_score = max(0.0, self._cusum_score + residual_filtre - CUSUM_K)

            # Mode gel : prédiction libre, sans correction par la mesure CGM.
            self.x = x_pred
            self.P = P_pred
        else:
            # Mode normal : correction Kalman avec la mesure CGM.
            self._residual_window = []
            self._cusum_score = 0.0
            self.x, self.P = self._kalman_correction(x_pred, P_pred, z)

        # 5) Alarme CUSUM confirmée
        slope = self._slope_cgm()
        # Décision Kalman-CUSUM : conformément au mémoire,
        # l'alerte est basée sur le dépassement du seuil CUSUM calibré.
        # La pente CGM n'est PAS une condition d'alerte Kalman ;
        # elle sert uniquement au dégel interne lorsque le CGM redescend.
        raw_alarm = (self._cusum_score > CUSUM_H)

        if just_degel or (not self.freeze):
            raw_alarm = False

        if raw_alarm:
            self._confirm_count += 1
        else:
            self._confirm_count = 0
            self._confirmed = False

        if self._confirm_count >= KALMAN_CONFIRM_STEPS:
            self._confirmed = True

        self._prev_cgm = cgm
        return p_libre, ecart_libre, self._cusum_score, self._confirmed

    def set_freeze(self, freeze):
        # Fusion passive : aucune commande externe du gel/dégel.
        pass

    def reset(self, cgm):
        cgm = float(cgm)
        self.x  = np.array([[cgm], [0.0]])
        self.P  = np.eye(2) * 10.0
        self.freeze = False
        self._residual_window = []
        self._cusum_score = 0.0
        self._confirm_count = 0
        self._confirmed = False
        self._neg_slope_count = 0
        self._cgm_history = [cgm]
        self._prev_cgm = cgm

CAS_KK_LL = "K_alerte  + L_alerte  → PANNE CONFIRMÉE"
CAS_KK_LS = "K_alerte  + L_silence → SUSPICION Kalman"
CAS_KS_LL = "K_silence + L_alerte  → SUSPICION LISA"
CAS_KS_LS = "K_silence + L_silence → OK"
CAS_SEQ   = "Alertes successives K→L ou L→K → SUSPICION SÉQUENTIELLE"


class FusionStateMachine:

    def __init__(self):
        self.etat      = "OK"
        self.confiance = None

        self.cmd_freeze = False
        self.cmd_reset  = False

        self.nb_panne         = 0
        self.nb_suspicion     = 0
        self.nb_suspicion_seq = 0
        self.nb_ok            = 0
        self.nb_panne_haute   = 0

        self.nb_cas = {
            CAS_KK_LL: 0,
            CAS_KK_LS: 0,
            CAS_KS_LL: 0,
            CAS_KS_LS: 0,
            CAS_SEQ:   0,
        }

        self._k_first_alert_step = None
        self._l_first_alert_step = None
        self.ecarts = []

        
        self._last_single_detector = None
        self._last_single_step     = None
        self._last_single_window   = None

    def _single_detector_courant(self, k_alert, l_alert):
        if k_alert and not l_alert:
            return "K"
        if l_alert and not k_alert:
            return "L"
        return None

    def _est_sequentielle_interbolus(self, detector_now, step, bolus_window_start):
        if detector_now not in ("K", "L"):
            return False
        if self._last_single_detector not in ("K", "L"):
            return False
        if detector_now == self._last_single_detector:
            return False
        if bolus_window_start is None or self._last_single_window is None:
            return False
        if bolus_window_start != self._last_single_window:
            return False
        return 0 < (step - self._last_single_step) <= SEQ_WINDOW_STEPS

    def update(self, k_alert, l_alert, step, cgm_val, sample_time, bolus_window_start=None):
        
        self.cmd_freeze = False
        self.cmd_reset  = False

        detector_now = self._single_detector_courant(k_alert, l_alert)

        if k_alert and l_alert:
            cas = CAS_KK_LL
            self.etat = "PANNE"
            self.confiance = "HAUTE"

            if self._k_first_alert_step is None:
                self._k_first_alert_step = step
            if self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._enregistrer_ecart(step, "HAUTE")

            
            self._last_single_detector = None
            self._last_single_step     = None
            self._last_single_window   = None

        elif detector_now and self._est_sequentielle_interbolus(detector_now, step, bolus_window_start):
            cas = CAS_SEQ
            self.etat = "SUSPICION_SEQUENTIELLE"
            self.confiance = "SEQUENTIELLE"

            if detector_now == "K" and self._k_first_alert_step is None:
                self._k_first_alert_step = step
            if detector_now == "L" and self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._enregistrer_ecart(step, "SEQUENTIELLE")

           
            self._last_single_detector = detector_now
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        elif k_alert and not l_alert:
            cas = CAS_KK_LS
            self.etat = "SUSPICION"
            self.confiance = None

            if self._k_first_alert_step is None:
                self._k_first_alert_step = step

            self._last_single_detector = "K"
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        elif l_alert and not k_alert:
            cas = CAS_KS_LL
            self.etat = "SUSPICION"
            self.confiance = None

            if self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._last_single_detector = "L"
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        else:
            cas = CAS_KS_LS
            self.etat = "OK"
            self.confiance = None
            self._reset_first_alerts()

        self.nb_cas[cas] += 1

        if self.etat == "PANNE":
            self.nb_panne += 1
            self.nb_panne_haute += 1
        elif self.etat == "SUSPICION_SEQUENTIELLE":
            self.nb_suspicion_seq += 1
        elif self.etat == "SUSPICION":
            self.nb_suspicion += 1
        else:
            self.nb_ok += 1

        return self.etat, self.confiance, cas

    def _enregistrer_ecart(self, step, confiance):
        if (self._k_first_alert_step is not None and
                self._l_first_alert_step is not None):
            ecart = self._l_first_alert_step - self._k_first_alert_step
            self.ecarts.append((step, ecart, confiance))
        elif self._k_first_alert_step is not None:
            self.ecarts.append((step, None, confiance))
        elif self._l_first_alert_step is not None:
            self.ecarts.append((step, None, confiance))

    def _reset_first_alerts(self):
        self._k_first_alert_step = None
        self._l_first_alert_step = None


# ============================================================
# GESTIONNAIRE DE MÉTRIQUES VP/FP/FN — LOGIQUE INTER-BOLUS (v5)
# ============================================================

class MetricsTracker:
    

    def __init__(self, name, steps_24h, mode="single"):
        self.name = name
        self.steps_24h = steps_24h
        self.mode = mode  

        self.vp = 0
        self.fp = 0
        self.fn = 0
        self.archives = []

        self._bolus_steps_sorted = []
        self._win_start = None
        self._win_end = None
        self._win_manque = False

        
        self._win_alerted = False

        self._win_k_alerted = False
        self._win_l_alerted = False
        self._win_simultaneous = False
        self._win_first_k = None
        self._win_first_l = None

    def register_bolus_steps(self, bolus_steps_dict):
    
        self._bolus_steps_sorted = sorted(bolus_steps_dict.keys())

    def _find_window(self, step):
       
        bs = self._bolus_steps_sorted
        if not bs or step < bs[0]:
            return None, None

        lo, hi = 0, len(bs) - 1
        idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if bs[mid] <= step:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        win_start = bs[idx]
        win_end = bs[idx + 1] if idx + 1 < len(bs) else None
        return win_start, win_end

    def _open_new_window(self, win_start, win_end, bolus_manque_ce_step):
        self._win_start = win_start
        self._win_end = win_end

   
        self._win_manque = bool(bolus_manque_ce_step)

        self._win_alerted = False
        self._win_k_alerted = False
        self._win_l_alerted = False
        self._win_simultaneous = False
        self._win_first_k = None
        self._win_first_l = None

    def update_single(self, step, alert, bolus_manque_ce_step):
        
        win_start, win_end = self._find_window(step)
        if win_start is None:
            return

        if win_start != self._win_start:
            if self._win_start is not None:
                self._close_window()
            self._open_new_window(win_start, win_end, bolus_manque_ce_step)

        if alert:
            self._win_alerted = True

        if (step > 0) and (step % self.steps_24h == 0):
            self._print_and_archive(step)
            self._reset_counters_only()

    def update_fusion(self, step, k_alert, l_alert, bolus_manque_ce_step):
       
        win_start, win_end = self._find_window(step)
        if win_start is None:
            return

        if win_start != self._win_start:
            if self._win_start is not None:
                self._close_window()
            self._open_new_window(win_start, win_end, bolus_manque_ce_step)

        if k_alert:
            self._win_k_alerted = True
            if self._win_first_k is None:
                self._win_first_k = step

        if l_alert:
            self._win_l_alerted = True
            if self._win_first_l is None:
                self._win_first_l = step

        if k_alert and l_alert:
            self._win_simultaneous = True

        if (step > 0) and (step % self.steps_24h == 0):
            self._print_and_archive(step)
            self._reset_counters_only()

    def update(self, step, alert, pump_failed=None, bolus_manque_ce_step=False):
        
        self.update_single(step, alert, bolus_manque_ce_step)

    def _window_detected(self):
        if self.mode == "fusion":
           
            return self._win_k_alerted and self._win_l_alerted
        return self._win_alerted

    def _close_window(self):
        detected = self._window_detected()

        if self._win_manque:
            if detected:
                self.vp += 1
            else:
                self.fn += 1
        else:
            
            if detected:
                self.fp += 1

    def _scores(self):
        precision = self.vp / max(1, self.vp + self.fp) * 100
        rappel = self.vp / max(1, self.vp + self.fn) * 100
        f1 = 2 * precision * rappel / max(0.01, precision + rappel)
        return precision, rappel, f1

    def _print_and_archive(self, step):
        precision, rappel, f1 = self._scores()
        sep = "─" * 72
        print(f"\n{'═'*72}")
        print(f"  MÉTRIQUES [{self.name}] — fenêtre 24h terminée au step {step}")
        print("  (unité : fenêtres inter-bolus)")
        print(sep)
        print(f"  {'VP (fenêtres détectées)':<30} {self.vp:>8}")
        print(f"  {'FP (fausses alarmes)':<30} {self.fp:>8}")
        print(f"  {'FN (fenêtres manquées)':<30} {self.fn:>8}")
        print(sep)
        print(f"  {'Précision':<30} {precision:>8.1f}%")
        print(f"  {'Rappel (sensibilité)':<30} {rappel:>8.1f}%")
        print(f"  {'Score F1':<30} {f1:>8.1f}%")
        print(f"{'═'*72}\n")
        self.archives.append({
            "window_end": step,
            "vp": self.vp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "rappel": rappel,
            "f1": f1,
        })

    def _reset_counters_only(self):
        self.vp = 0
        self.fp = 0
        self.fn = 0
      

    def resume_final(self, step):
       
        if (self._win_start is not None and
                self._win_end is not None and
                step >= self._win_end):
            self._close_window()
            self._win_start = None

        precision, rappel, f1 = self._scores()
        sep = "─" * 72
        print()
        print("═" * 72)
        print(f"  MÉTRIQUES FINALES [{self.name}] — arrêt au step {step}")
        print("  (unité : fenêtres inter-bolus)")
        print(sep)
        print(f"  {'VP (fenêtres détectées)':<30} {self.vp:>8}")
        print(f"  {'FP (fausses alarmes)':<30} {self.fp:>8}")
        print(f"  {'FN (fenêtres manquées)':<30} {self.fn:>8}")
        print(sep)
        print(f"  {'Précision':<30} {precision:>8.1f}%")
        print(f"  {'Rappel (sensibilité)':<30} {rappel:>8.1f}%")
        print(f"  {'Score F1':<30} {f1:>8.1f}%")
        print("═" * 72)
        print()




def _calc_metrics_interbolus_final(name, bolus_steps_dict, missed_bolus_history,
                                  k_alert_history=None, l_alert_history=None,
                                  single_alert_history=None, n_steps=0):
    """
    Calcul final robuste par fenêtres inter-bolus complètes.

    Règles appliquées :
      - 1 bolus manqué = 1 événement panne ;
      - fenêtre évaluée : [bolus_k, bolus_{k+1}[ ;
      - VP : bolus_k manqué et détection avant bolus suivant ;
      - FN : bolus_k manqué et aucune détection avant bolus suivant ;
      - FP : bolus_k injecté et détection dans la fenêtre ;
      - Fusion : détection seulement si Kalman ET LISA alertent dans la même fenêtre,
                 simultanément ou séquentiellement.

    Important : la dernière fenêtre incomplète n'est pas comptée.
    """
    bolus_steps = sorted(s for s in bolus_steps_dict.keys() if s < n_steps)

    vp = fp = fn = 0
    details = []

    for i in range(len(bolus_steps) - 1):
        bs = bolus_steps[i]
        be = bolus_steps[i + 1]

        if bs >= n_steps or be > n_steps:
            continue

        missed = bool(missed_bolus_history[bs]) if bs < len(missed_bolus_history) else False

        if single_alert_history is not None:
            detected = bool(np.any(single_alert_history[bs:be]))
            k_seen = l_seen = None
        else:
            k_seen = bool(np.any(k_alert_history[bs:be]))
            l_seen = bool(np.any(l_alert_history[bs:be]))
            detected = k_seen and l_seen

        if missed:
            if detected:
                vp += 1
                cls = "VP"
            else:
                fn += 1
                cls = "FN"
        else:
            if detected:
                fp += 1
                cls = "FP"
            else:
                cls = "TN"

        details.append((bs, be, missed, detected, cls, k_seen, l_seen))

    precision = vp / max(1, vp + fp) * 100
    rappel = vp / max(1, vp + fn) * 100
    f1 = 2 * precision * rappel / max(0.01, precision + rappel)

    sep = "─" * 72
    print()
    print("═" * 72)
    print(f"  MÉTRIQUES FINALES ROBUSTES [{name}] — durée simulée")
    print("  (unité : fenêtres inter-bolus complètes)")
    print(sep)
    print(f"  {'VP (fenêtres détectées)':<30} {vp:>8}")
    print(f"  {'FP (fausses alarmes)':<30} {fp:>8}")
    print(f"  {'FN (fenêtres manquées)':<30} {fn:>8}")
    print(sep)
    print(f"  {'Précision':<30} {precision:>8.1f}%")
    print(f"  {'Rappel (sensibilité)':<30} {rappel:>8.1f}%")
    print(f"  {'Score F1':<30} {f1:>8.1f}%")
    print("═" * 72)

    return {
        "name": name,
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "rappel": rappel,
        "f1": f1,
        "details": details,
    }


def afficher_resume_metriques_24h(bolus_steps_dict, missed_bolus_history,
                                  k_alert_history, l_alert_history, n_steps):
    print(f"\n{'='*72}")
    print("  MÉTRIQUES FINALES — RÉSUMÉ DES 3 DÉTECTEURS")
    print("  Logique stricte : 1 bolus manqué = 1 fenêtre à détecter avant le bolus suivant")
    print(f"  Durée évaluée : {n_steps * sample_time / 60:.1f} h")
    print(f"{'='*72}")

    r_k = _calc_metrics_interbolus_final(
        "KALMAN", bolus_steps_dict, missed_bolus_history,
        single_alert_history=np.array(k_alert_history, dtype=bool),
        n_steps=n_steps,
    )

    r_l = _calc_metrics_interbolus_final(
        "LISA", bolus_steps_dict, missed_bolus_history,
        single_alert_history=np.array(l_alert_history, dtype=bool),
        n_steps=n_steps,
    )

    r_f = _calc_metrics_interbolus_final(
        "FUSION", bolus_steps_dict, missed_bolus_history,
        k_alert_history=np.array(k_alert_history, dtype=bool),
        l_alert_history=np.array(l_alert_history, dtype=bool),
        n_steps=n_steps,
    )

    return r_k, r_l, r_f

# ============================================================
# INITIALISATION PATIENT & ENVIRONNEMENT
# ============================================================

patient    = T1DPatient.withName(PATIENT_NAME)
sensor     = CGMSensor.withName("Dexcom", seed=1)
pump       = InsulinPump.withName("Insulet")
start_time = datetime.now()

env_tmp     = T1DSimEnv(patient=patient, sensor=sensor, pump=pump,
                        scenario=AdultScenario(start_time, [], 3))
sample_time = env_tmp.sample_time
del env_tmp

REPAS    = build_repas_for_horizon(sample_time, horizon_steps=10000)
patient  = T1DPatient.withName(PATIENT_NAME)
sensor   = CGMSensor.withName("Dexcom", seed=1)
pump     = InsulinPump.withName("Insulet")
scenario = AdultScenario(start_time, REPAS, sample_time)
env      = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

for r in REPAS:
    r["step"] = int(r["min"] / sample_time)

BOLUS_CARBS_STEPS = {max(0, r["step"] - PREBOLUS_STEPS): r["carbs"] for r in REPAS}
BOLUS_STEPS       = BOLUS_CARBS_STEPS


def get_inter_bolus_window_start(step, bolus_steps_dict):
    bs = sorted(bolus_steps_dict.keys())
    if not bs or step < bs[0]:
        return None
    lo, hi, idx = 0, len(bs) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if bs[mid] <= step: idx = mid; lo = mid + 1
        else:               hi  = mid - 1
    return bs[idx]


def update_bolus_steps(current_step, scenario, sample_time):
    scenario.extend_if_needed(current_step, sample_time, horizon=500)
    for r in scenario.repas:
        s_r = r.get("step")
        if s_r is None: continue
        bolus_step = max(0, s_r - PREBOLUS_STEPS)
        if bolus_step not in BOLUS_CARBS_STEPS:
            BOLUS_CARBS_STEPS[bolus_step] = r["carbs"]


STEPS_24H = int((24 * 60) / sample_time)
MAX_SIM_STEPS = STEPS_24H  # 24 h exactes

print(f"\n{'='*72}")
print(f"  FUSION KALMAN + LISA v1 — Détection de panne pompe ({PATIENT_NAME})")
print(f"{'='*72}")
print(f"  Sample time      : {sample_time} min")
print(f"  LISA LW / SW     : {LISA_LW} steps ({LISA_LW*sample_time:.0f} min) / "
      f"{LISA_SW} steps ({LISA_SW*sample_time:.0f} min)")
print(f"  LISA GFM seuil   : {LISA_GFM_THRESHOLD}")
print(f"  LISA IFM seuil   : {LISA_IFM_THRESHOLD}")
print(f"  LISA Pente seuil : {LISA_SLOPE_THRESHOLD} mg/dL/min")
print(f"  LISA Score min   : {LISA_SCORE_MIN} / 4 pts  (GFM×2 + IFM×1 + Pente×1)")
print(f"  LISA Confirmation: {LISA_CONFIRMATION_STEPS} steps = {LISA_CONFIRMATION_STEPS*sample_time:.0f} min")
print(f"  Kalman CUSUM     : k={CUSUM_K} | h={CUSUM_H} | résidus fenêtre={CUSUM_RES_WINDOW} steps")
print(f"  Kalman confirm   : {KALMAN_CONFIRM_STEPS} steps | décision : CUSUM > seuil | dégel={KalmanDetector.N_DEGEL} steps descendants")
print(f"  Cycle pannes     : {PANNE_DUREE} steps HS ({PANNE_DUREE*sample_time:.0f} min) → "
      f"{MARCHE_DUREE} steps OK ({MARCHE_DUREE*sample_time:.0f} min)")
print(f"  Repas            : {len(REPAS_JOURNEE)} repas/jour, cycle {MINUTES_PAR_JOUR} min")
print(f"  Bolus            : ICR={ICR}g/U | ISF={ISF}mg/dL/U")
print("  Durée évaluée    : 24 h exactes")
print("  Arrêt            : automatique après 24 h")
print(f"{'='*72}\n")

print(f"{'Step':>5} {'Heure':>6} {'CGM':>7} "
      f"{'K_alr':>7} {'L_alr':>7} {'Score':>8} {'Consec':>7} "
      f"{'GFM':>10} {'IFM':>7} {'Pente':>7} "
      f"{'Fusion':>22} {'Confiance':>14} {'Détecteur':>15} "
      f"{'Cas':>42} {'Réel':>8}")
print("─" * 170)


# ============================================================
# GRAPHE — 4 panneaux
# ============================================================

plt.style.use("default")

C_BG        = "white"
C_GRID      = "#cccccc"
C_SPINE     = "#aaaaaa"
C_TICK      = "#333333"
C_CGM       = "#005f5f"      # vert-bleu pour distinguer de l'enfant (bleu)
C_PRED      = "#b35900"
C_KSCORE    = "#8b0000"
C_GFM       = "#4b0082"
C_SLOPE     = "#005f5f"
C_IFM       = "#7a3500"
C_OR_BRUT   = "#aaaaaa"
C_SEUIL_K   = "#cc0000"
C_SEUIL_GFM = "#4b0082"
C_SEUIL_SLP = "#005f5f"
C_SEUIL_IFM = "#7a3500"
C_MEAL      = "#cc5500"
C_BOLUS_OK  = "#003f8a"
C_BOLUS_FAIL= "#cc0000"
C_HAUTE     = "#cc0000"
C_SUSPICION = "#005f5f"
C_SEQ       = "#b87700"
C_BAND_FAIL = "#ffcccc"
C_BAND_OK   = "#e6f7ee"
C_ZONE_CIBLE= "#e8f4fd"

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(C_BG)

gs = gridspec.GridSpec(4, 1, figure=fig,
                       height_ratios=[3, 1, 1, 1], hspace=0.08)

ax_cgm = fig.add_subplot(gs[0])
ax_k   = fig.add_subplot(gs[1], sharex=ax_cgm)
ax_l   = fig.add_subplot(gs[2], sharex=ax_cgm)
ax_fus = fig.add_subplot(gs[3], sharex=ax_cgm)
AXES   = [ax_cgm, ax_k, ax_l, ax_fus]

for ax in AXES:
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)
    ax.tick_params(colors=C_TICK, labelsize=7)
    for s in ["top", "right"]:   ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color(C_SPINE)

# Panneau CGM
line_cgm,  = ax_cgm.plot([], [], color=C_CGM,  lw=2.5, zorder=3)
line_pred, = ax_cgm.plot([], [], color=C_PRED, ls="--", lw=1.5, zorder=3, alpha=0.85)
sc_meals       = ax_cgm.scatter([], [], marker="^", s=160, color=C_MEAL,       zorder=6)
sc_bolus_ok    = ax_cgm.scatter([], [], marker="o", s=120, color=C_BOLUS_OK,   zorder=6)
sc_bolus_fail  = ax_cgm.scatter([], [], marker="x", s=160, color=C_BOLUS_FAIL, zorder=6, lw=2.5)
sc_panne_haute = ax_cgm.scatter([], [], marker="D", s=180, color=C_HAUTE,      zorder=9, lw=1.5)
sc_suspicion   = ax_cgm.scatter([], [], marker="o", s=80,  color=C_SUSPICION,  zorder=8,
                                facecolors="none", lw=2.0)
sc_susp_seq    = ax_cgm.scatter([], [], marker="s", s=100, color=C_SEQ,        zorder=8,
                                facecolors="none", lw=2.0)

ax_cgm.axhline(70,  color=C_BOLUS_OK,   ls="--", lw=1.2, alpha=0.7)
ax_cgm.axhline(180, color=C_BOLUS_FAIL, ls="--", lw=1.2, alpha=0.7)
ax_cgm.axhspan(70, 180, alpha=0.08, color=C_ZONE_CIBLE)
ax_cgm.set_ylim(40, 450)
ax_cgm.set_ylabel("Glycémie (mg/dL)", color=C_TICK, fontsize=8)

legend_elems = [
    Line2D([0],[0], color=C_CGM,   lw=2.5,          label="CGM"),
    Line2D([0],[0], color=C_PRED,  lw=1.5, ls="--",  label="Prédiction Kalman"),
    Patch(facecolor=C_BAND_FAIL, alpha=0.5,           label="Pompe HS (réel)"),
    Patch(facecolor=C_BAND_OK,   alpha=0.5,           label="Pompe OK (réel)"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor=C_MEAL,     ms=8, ls="None", label="Repas"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BOLUS_OK, ms=7, ls="None", label="Bolus injecté"),
    Line2D([0],[0], marker="x", color=C_BOLUS_FAIL, ms=8, ls="None",  label="Bolus manqué"),
    Line2D([0],[0], marker="D", color=C_HAUTE,      ms=8, ls="None",  label="PANNE CONFIRMÉE (K+L)"),
    Line2D([0],[0], marker="o", color=C_SUSPICION,  ms=8, ls="None",
           markerfacecolor="none", label="SUSPICION (1 détecteur)"),
    Line2D([0],[0], marker="s", color=C_SEQ,        ms=8, ls="None",
           markerfacecolor="none", label="SUSPICION SÉQUENTIELLE"),
]
ax_cgm.legend(handles=legend_elems, loc="upper left",
              fontsize=7, facecolor="white", edgecolor=C_SPINE, ncol=3)

# Panneau Kalman
line_kscore, = ax_k.plot([], [], color=C_KSCORE, lw=2.0)
ax_k.axhline(CUSUM_H, color=C_SEUIL_K, ls="--", lw=1.2, alpha=0.85)
ax_k.set_ylim(0, max(100, CUSUM_H * 1.4))
ax_k.set_ylabel("Score\nCUSUM", color=C_TICK, fontsize=7)

# Panneau LISA
line_gfm,    = ax_l.plot([], [], color=C_GFM,    lw=2.0,       label="GFM/100")
line_ifm,    = ax_l.plot([], [], color=C_IFM,    lw=1.5, ls="-.", label="IFM")
line_slope,  = ax_l.plot([], [], color=C_SLOPE,   lw=1.5,       label="Pente")
line_consec, = ax_l.plot([], [], color=C_OR_BRUT, lw=1.0, ls=":",
                          label=f"Consécutifs/{LISA_CONFIRMATION_STEPS}")

ax_l.axhline(LISA_GFM_THRESHOLD / 100, color=C_SEUIL_GFM, ls="--", lw=1.2, alpha=0.85)
ax_l.axhline(LISA_IFM_THRESHOLD,        color=C_SEUIL_IFM, ls="--", lw=1.2, alpha=0.85)
ax_l.axhline(LISA_SLOPE_THRESHOLD,      color=C_SEUIL_SLP, ls="--", lw=1.2, alpha=0.85)
ax_l.axhline(1.0, color=C_OR_BRUT, ls=":", lw=1.0, alpha=0.6, label="Seuil confirmation")
ax_l.set_ylim(-2, 12)
ax_l.set_ylabel("Indicateurs\nLISA", color=C_TICK, fontsize=7)
ax_l.legend(loc="upper left", fontsize=6, facecolor="white", edgecolor=C_SPINE, ncol=2)

# Panneau fusion
ax_fus.set_ylim(-0.5, 3.5)
ax_fus.set_yticks([0, 1, 1.5, 2, 3])
ax_fus.set_yticklabels(["OK", "SUSP.", "SUSP.\nSÉQ.", "PANNE\nfusion", "PANNE\nréelle"],
                        color=C_TICK, fontsize=6)
ax_fus.set_ylabel("Décision", color=C_TICK, fontsize=7)
ax_fus.set_xlabel("Heure simulée", color=C_TICK, fontsize=8)

fusion_level_history = []
real_level_history   = []

txt_etat  = ax_cgm.text(0.99, 0.97, "", transform=ax_cgm.transAxes,
                         color=C_CGM, fontsize=13, ha="right", va="top",
                         fontfamily="monospace", fontweight="bold")
txt_info  = ax_cgm.text(0.99, 0.89, "", transform=ax_cgm.transAxes,
                         color="#444444", fontsize=8, ha="right", va="top")
txt_cas   = ax_cgm.text(0.99, 0.82, "", transform=ax_cgm.transAxes,
                         color="#7a5c00", fontsize=8, ha="right", va="top",
                         fontfamily="monospace")
txt_alarm = ax_cgm.text(0.5, 0.50, "", transform=ax_cgm.transAxes,
                         color=C_HAUTE, fontsize=12, ha="center", va="center",
                         fontfamily="monospace", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe5e5",
                                   edgecolor=C_HAUTE, alpha=0.0))

fig.suptitle(
    f"Fusion Kalman + LISA v1 — Détection de panne pompe ({PATIENT_NAME})\n"
    f"LISA: score≥{LISA_SCORE_MIN} + confirmation {LISA_CONFIRMATION_STEPS} steps "
    f"({LISA_CONFIRMATION_STEPS*3}min) | Kalman-CUSUM: k={CUSUM_K:.1f}, h={CUSUM_H:.1f}  —  24h auto",
    fontsize=10, color="#111111", fontfamily="monospace"
)

failure_bands = {ax: [] for ax in AXES}
fill_ref      = [None]

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.ion()
plt.show()


# ============================================================
# INITIALISATION BOUCLE
# ============================================================

s    = env.reset()
obs  = s.observation
done = s.done

index      = 0
prev_state = None

kalman = KalmanDetector(float(obs.CGM), sample_time)
lisa   = LISADetector(sample_time)
fusion = FusionStateMachine()

metrics_kalman = MetricsTracker("KALMAN", STEPS_24H)
metrics_lisa   = MetricsTracker("LISA",   STEPS_24H)
metrics_fusion = MetricsTracker("FUSION", STEPS_24H, mode="fusion")

metrics_kalman.register_bolus_steps(BOLUS_CARBS_STEPS)
metrics_lisa.register_bolus_steps(BOLUS_CARBS_STEPS)
metrics_fusion.register_bolus_steps(BOLUS_CARBS_STEPS)

times             = []
cgm_vals          = []
pred_vals         = []
pump_states       = []
k_scores          = []
k_alerts_h        = []
l_gfm_h           = []
l_ifm_h           = []
l_slope_h         = []
l_consec_h        = []
l_or_brut_h       = []
l_alerts_h        = []
missed_bolus_h    = []
fusion_states     = []
cas_history       = []
confiance_history = []

MAX_PTS = 80

bolus_total_prevu   = 0
bolus_total_injecte = 0
bolus_total_manque  = 0
insuline_injectee_U = 0.0
insuline_manquee_U  = 0.0
insuline_basale_U   = 0.0


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

STOP_REASON = "Ctrl+C"

try:
    while index < MAX_SIM_STEPS:
        update_bolus_steps(index, scenario, sample_time)
        metrics_kalman.register_bolus_steps(BOLUS_CARBS_STEPS)
        metrics_lisa.register_bolus_steps(BOLUS_CARBS_STEPS)
        metrics_fusion.register_bolus_steps(BOLUS_CARBS_STEPS)

        cgm_val    = float(obs.CGM)
        sim_min    = index * sample_time
        time_label = (start_time + timedelta(minutes=sim_min)).strftime("%H:%M")

        carbs_for_bolus = BOLUS_CARBS_STEPS.get(index, 0)
        bolus_plan      = calculer_bolus(carbs_for_bolus, cgm_val) if carbs_for_bolus > 0 else 0.0
        failed          = pump_is_failed(index)

        if prev_state is not None and failed != prev_state:
            sep = "─" * 110
            if failed:
                print(f"\n{sep}\n  [!] PANNE POMPE RÉELLE — step {index} ({time_label})\n{sep}\n")
            else:
                print(f"\n{sep}\n  [v] POMPE RÉTABLIE — step {index} ({time_label})\n{sep}\n")
        prev_state = failed

        # ── Basal dynamique adulte ───────────────────────────────────────────
        dynamic_basal = BASAL
        if cgm_val > 170: dynamic_basal += 0.008
        if cgm_val > 210: dynamic_basal += 0.015
        if cgm_val < 100: dynamic_basal -= 0.008
        if cgm_val <  85: dynamic_basal -= 0.015
        if cgm_val <  70: dynamic_basal = 0.0
        dynamic_basal = max(0.0, dynamic_basal)

        if bolus_plan == 0.0 and cgm_val > 200.0:
            bolus_final = min(4.0, max(0.0, (cgm_val - CGM_CIBLE) / ISF))
        else:
            bolus_final = bolus_plan

        action = (Action(basal=0.0, bolus=0.0) if failed else
                  Action(basal=dynamic_basal, bolus=bolus_final))

        insulin_dose_step = (0.0 if failed else
                             dynamic_basal * sample_time + bolus_final)

        # ── Kalman ──────────────────────────────────────────────────────────
        pred, residual, k_score, k_alert = kalman.update(cgm_val)

        # ── LISA ────────────────────────────────────────────────────────────
        gfm, ifm, slope, l_alert, or_brut, consec = lisa.update(cgm_val, insulin_dose_step)

        # ── Fusion ──────────────────────────────────────────────────────────
        bolus_window_start = get_inter_bolus_window_start(index, BOLUS_CARBS_STEPS)
        etat_fusion, confiance, cas = fusion.update(
            k_alert, l_alert, index, cgm_val, sample_time, bolus_window_start
        )

        # ── Détecteur responsable ────────────────────────────────────────────
        if k_alert and l_alert:    detecteur_txt = "Kalman + LISA"
        elif k_alert:              detecteur_txt = "Kalman seul"
        elif l_alert:              detecteur_txt = "LISA seule"
        else:                      detecteur_txt = "Aucun"

        # ── Notifications ────────────────────────────────────────────────────
        if etat_fusion == "PANNE":
            notifier_panne(index)
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":
            _ecart_seq = 0
            if (fusion._k_first_alert_step is not None and
                    fusion._l_first_alert_step is not None):
                _ecart_seq = abs(fusion._l_first_alert_step - fusion._k_first_alert_step)
            notifier_suspicion_sequentielle(index, _ecart_seq, sample_time)
        elif etat_fusion == "SUSPICION":
            notifier_suspicion(index)

        # ── Commandes Kalman depuis la fusion ────────────────────────────────
        # Fusion passive : on ne commande ni Kalman ni LISA.
        # Les deux détecteurs restent indépendants.
        # Ancienne logique supprimée :
        #   - pas de kalman.reset(cgm_val) demandé par la fusion
        #   - pas de lisa.reset() demandé par la fusion
        #   - pas de kalman.set_freeze(...) demandé par la fusion

        # ── Métriques inter-bolus ─────────────────────────────────────────────
        bolus_attendu_ce_step = index in BOLUS_CARBS_STEPS
        bolus_manque_ce_step  = failed and bolus_attendu_ce_step
        missed_bolus_h.append(bolus_manque_ce_step)

        metrics_kalman.update(index, k_alert, bolus_manque_ce_step=bolus_manque_ce_step)
        metrics_lisa.update(index, l_alert, bolus_manque_ce_step=bolus_manque_ce_step)
        metrics_fusion.update_fusion(index, k_alert, l_alert, bolus_manque_ce_step)

        # ── Console ──────────────────────────────────────────────────────────
        conf_txt = confiance if confiance else "—"
        or_txt   = "OR" if or_brut else "  "
        print(
            f"[{index:>4}] {time_label} "
            f"{cgm_val:>7.1f} "
            f"{'OUI' if k_alert else 'NON':>7} "
            f"{'OUI' if l_alert else 'NON':>7} "
            f"{or_txt:>8} {consec:>7} "
            f"{gfm:>10.1f} "
            f"{ifm:>7.3f} "
            f"{slope:>7.2f} "
            f"{etat_fusion:>22} "
            f"{conf_txt:>14} "
            f"{detecteur_txt:>15} "
            f"{'[B=' + str(bolus_plan) + 'U]' if bolus_plan > 0 else '':>8} "
            f"{cas:>42} "
            f"{'PANNE' if failed else 'OK':>8}"
        )

        # ── Historiques ───────────────────────────────────────────────────────
        times.append(time_label)
        cgm_vals.append(cgm_val)
        pred_vals.append(pred)
        pump_states.append(failed)
        k_scores.append(k_score)
        k_alerts_h.append(k_alert)
        l_gfm_h.append(gfm)
        l_ifm_h.append(ifm)
        l_slope_h.append(slope)
        l_consec_h.append(consec)
        l_or_brut_h.append(or_brut)
        l_alerts_h.append(l_alert)
        fusion_states.append(etat_fusion)
        cas_history.append(cas)
        confiance_history.append(confiance)
        fusion_level_history.append(
            2   if etat_fusion == "PANNE" else
            1.5 if etat_fusion == "SUSPICION_SEQUENTIELLE" else
            1   if etat_fusion == "SUSPICION" else 0
        )
        real_level_history.append(3 if failed else 2.85)

        # ── Suivi insuline ────────────────────────────────────────────────────
        if bolus_attendu_ce_step:
            bolus_total_prevu += 1
            if failed:
                bolus_total_manque  += 1
                insuline_manquee_U  += bolus_plan
            else:
                bolus_total_injecte += 1
                insuline_injectee_U += bolus_final
        elif not failed and bolus_final > 0:
            insuline_injectee_U += bolus_final
        if not failed:
            insuline_basale_U += dynamic_basal * sample_time

        # ── Affichage graphique ───────────────────────────────────────────────
        disp_times         = times[-MAX_PTS:]
        disp_cgm           = cgm_vals[-MAX_PTS:]
        disp_pred          = pred_vals[-MAX_PTS:]
        disp_states        = pump_states[-MAX_PTS:]
        disp_kscore        = k_scores[-MAX_PTS:]
        disp_gfm           = l_gfm_h[-MAX_PTS:]
        disp_ifm           = l_ifm_h[-MAX_PTS:]
        disp_slope         = l_slope_h[-MAX_PTS:]
        disp_consec        = l_consec_h[-MAX_PTS:]
        disp_fus           = fusion_level_history[-MAX_PTS:]
        disp_real          = real_level_history[-MAX_PTS:]
        disp_fusion_states = fusion_states[-MAX_PTS:]
        disp_conf          = confiance_history[-MAX_PTS:]

        x      = list(range(len(disp_times)))
        offset = max(0, len(times) - MAX_PTS)

        line_cgm.set_data(x, disp_cgm)
        line_pred.set_data(x, disp_pred)
        line_kscore.set_data(x, disp_kscore)
        line_gfm.set_data(x, [v / 100.0 for v in disp_gfm])
        line_ifm.set_data(x, disp_ifm)
        line_slope.set_data(x, disp_slope)
        line_consec.set_data(x, [v / max(1, LISA_CONFIRMATION_STEPS) for v in disp_consec])

        if fill_ref[0] is not None:
            fill_ref[0].remove()
        fill_ref[0] = ax_cgm.fill_between(x, disp_cgm, alpha=0.08, color=C_CGM, zorder=2)

        # Panneau fusion — barres colorées
        ax_fus.cla()
        ax_fus.set_facecolor(C_BG)
        ax_fus.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)
        ax_fus.set_ylim(-0.5, 3.5)
        ax_fus.set_yticks([0, 1, 1.5, 2, 3])
        ax_fus.set_yticklabels(["OK", "SUSP.", "SUSP.\nSEQ.", "PANNE\nfusion", "PANNE\nréelle"],
                                color=C_TICK, fontsize=6)
        ax_fus.set_ylabel("Décision", color=C_TICK, fontsize=7)
        ax_fus.set_xlabel("Heure simulée", color=C_TICK, fontsize=8)
        ax_fus.tick_params(colors=C_TICK, labelsize=7)
        for s_name in ["top", "right"]:   ax_fus.spines[s_name].set_visible(False)
        for s_name in ["bottom", "left"]: ax_fus.spines[s_name].set_color(C_SPINE)

        for i_f, (fv, rv, fc) in enumerate(zip(disp_fus, disp_real, disp_conf)):
            if fv == 2:     color_f = C_HAUTE
            elif fv == 1.5: color_f = C_SEQ
            elif fv == 1:   color_f = "#e6a817"
            else:           color_f = "#c8e6c9"
            ax_fus.bar(i_f, fv + 0.5, bottom=-0.5, width=1.0,
                       color=color_f, alpha=0.80, align="edge")
            ax_fus.plot(i_f + 0.5, rv, marker="_", ms=10,
                        color=C_BOLUS_FAIL, lw=2, alpha=0.9)

        for ax in AXES:
            ax.set_xlim(-1, len(disp_times))

        ticks = list(range(0, len(disp_times), 6))
        ax_fus.set_xticks(ticks)
        ax_fus.set_xticklabels(
            [disp_times[i] for i in ticks if i < len(disp_times)],
            fontsize=7, color="#64748b"
        )
        for ax in [ax_cgm, ax_k, ax_l]:
            ax.set_xticks(ticks)
            ax.set_xticklabels([], visible=False)

        # Bandes panne/marche
        for ax in AXES:
            for band in failure_bands[ax]:
                try: band.remove()
                except Exception: pass
            failure_bands[ax].clear()

        i_seg = 0
        while i_seg < len(disp_states):
            seg_state = disp_states[i_seg]; j_seg = i_seg
            while j_seg < len(disp_states) and disp_states[j_seg] == seg_state:
                j_seg += 1
            c_band = C_BAND_FAIL if seg_state else C_BAND_OK
            a_band = 0.45 if seg_state else 0.35
            for ax in AXES:
                b = ax.axvspan(i_seg - 0.5, j_seg - 0.5, alpha=a_band, color=c_band, zorder=1)
                failure_bands[ax].append(b)
            i_seg = j_seg

        # Repas
        mx, my = [], []
        for r in scenario.repas:
            r_step = r.get("step")
            if r_step is None: continue
            vi = r_step - offset
            if 0 <= vi < len(disp_times):
                mx.append(vi); my.append(disp_cgm[vi] + 18)
        sc_meals.set_offsets(np.c_[mx, my] if mx else np.empty((0, 2)))

        # Bolus
        ok_x, ok_y, fx, fy = [], [], [], []
        for step_b in BOLUS_STEPS:
            vi = step_b - offset
            if 0 <= vi < len(disp_times):
                if pump_is_failed(step_b): fx.append(vi); fy.append(disp_cgm[vi] - 18)
                else:                      ok_x.append(vi); ok_y.append(disp_cgm[vi] - 18)
        sc_bolus_ok.set_offsets(np.c_[ok_x, ok_y] if ok_x else np.empty((0, 2)))
        sc_bolus_fail.set_offsets(np.c_[fx, fy] if fx else np.empty((0, 2)))

        # Alertes
        ph_x, ph_y, su_x, su_y, sq_x, sq_y = [], [], [], [], [], []
        for i_a, (fs, fc) in enumerate(zip(disp_fusion_states, disp_conf)):
            if fs == "PANNE":
                ph_x.append(i_a); ph_y.append(disp_cgm[i_a] + 30)
            elif fs == "SUSPICION_SEQUENTIELLE":
                sq_x.append(i_a); sq_y.append(disp_cgm[i_a] + 25)
            elif fs == "SUSPICION":
                su_x.append(i_a); su_y.append(disp_cgm[i_a] + 20)
        sc_panne_haute.set_offsets(np.c_[ph_x, ph_y] if ph_x else np.empty((0, 2)))
        sc_susp_seq.set_offsets(np.c_[sq_x, sq_y] if sq_x else np.empty((0, 2)))
        sc_suspicion.set_offsets(np.c_[su_x, su_y] if su_x else np.empty((0, 2)))

        # Textes
        if etat_fusion == "PANNE":                     col_etat = C_HAUTE
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":  col_etat = C_SEQ
        elif etat_fusion == "SUSPICION":               col_etat = "#e6a817"
        else:                                          col_etat = C_CGM
        txt_etat.set_text(f"{cgm_val:.0f} mg/dL  [{etat_fusion}]")
        txt_etat.set_color(col_etat)

        txt_info.set_text(
            f"{time_label} | step {index} | "
            f"Kalman {'✓' if k_alert else '·'}  LISA {'✓' if l_alert else '·'} "
            f"(Score≥{LISA_SCORE_MIN}: {'✓' if or_brut else '·'} ×{consec}/{LISA_CONFIRMATION_STEPS})  |  "
            f"GFM={gfm:.1f}  IFM={ifm:.3f}  Pente={slope:.2f}  |  "
            f"Réel : {'PANNE' if failed else 'OK'}"
        )

        ecart_txt = ""
        if fusion.ecarts:
            last_step, last_ecart, last_conf = fusion.ecarts[-1]
            if last_ecart is not None:
                signe     = "+" if last_ecart > 0 else ""
                ecart_txt = (f"Écart K→L : {signe}{last_ecart} steps "
                             f"({signe}{last_ecart*sample_time:.0f} min)")
            else:
                ecart_txt = f"Un seul détecteur ({last_conf})"
        txt_cas.set_text(f"Cas : {cas}\nConfiance : {confiance if confiance else '—'}\n{ecart_txt}")

        if etat_fusion == "PANNE":
            txt_alarm.set_text("PANNE CONFIRMÉE — Kalman + LISA")
            txt_alarm.set_color(C_HAUTE)
            txt_alarm.get_bbox_patch().set_edgecolor(C_HAUTE)
            txt_alarm.get_bbox_patch().set_facecolor("#ffe5e5")
            txt_alarm.get_bbox_patch().set_alpha(0.90)
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":
            if (fusion._k_first_alert_step is not None and
                    fusion._l_first_alert_step is not None):
                _ordre = "K → L" if fusion._k_first_alert_step <= fusion._l_first_alert_step else "L → K"
            else:
                _ordre = "K ↔ L"
            txt_alarm.set_text(f"SUSPICION SÉQUENTIELLE — {_ordre}")
            txt_alarm.set_color(C_SEQ)
            txt_alarm.get_bbox_patch().set_edgecolor(C_SEQ)
            txt_alarm.get_bbox_patch().set_facecolor("#fff3cd")
            txt_alarm.get_bbox_patch().set_alpha(0.90)
        elif etat_fusion == "SUSPICION":
            txt_alarm.set_text(f"SUSPICION — {detecteur_txt}")
            txt_alarm.set_color("#7a5c00")
            txt_alarm.get_bbox_patch().set_edgecolor("#e6a817")
            txt_alarm.get_bbox_patch().set_facecolor("#fff8e1")
            txt_alarm.get_bbox_patch().set_alpha(0.85)
        else:
            txt_alarm.set_text("")
            txt_alarm.get_bbox_patch().set_alpha(0.0)

        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass

        if STEP_DELAY > 0:
            time.sleep(STEP_DELAY)

        if done:
            print(f"\n[INFO] Reset environnement (step {index})\n")
            p2  = T1DPatient.withName(PATIENT_NAME)
            s2  = CGMSensor.withName("Dexcom", seed=index)
            pu2 = InsulinPump.withName("Insulet")
            sc2 = scenario
            sc2.start_time = start_time + timedelta(minutes=index * sample_time)
            env2 = T1DSimEnv(patient=p2, sensor=s2, pump=pu2, scenario=sc2)
            s    = env2.reset()
            env  = env2
            kalman = KalmanDetector(float(s.observation.CGM), sample_time)
            lisa   = LISADetector(sample_time)
        else:
            s = env.step(action)

        obs  = s.observation
        done = s.done
        index += 1

    STOP_REASON = f"fin automatique {SIM_DURATION_HOURS} h"
    raise KeyboardInterrupt


except KeyboardInterrupt:
    afficher_resume_metriques_24h(
        BOLUS_CARBS_STEPS,
        missed_bolus_h,
        k_alerts_h,
        l_alerts_h,
        index,
    )

plt.ioff()