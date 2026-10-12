# ─────────────────────────────────────────────────────────────────────────────

#
# SYSTÈME : Fusion Kalman + LISA pour la détection de panne de pompe à insuline
# PROFIL   : Adolescent #001 (SimGlucose / UVa-Padova)

# ─────────────────────────────────────────────────────────────────────────────

# ─── IMPORTS ─────────────────────────────────────────────────────────────────

import time                        # Pour les pauses entre steps (STEP_DELAY)
import threading                   # Pour envoyer les notifications ntfy en arrière-plan
import urllib.request              # Pour effectuer les requêtes HTTP vers ntfy.sh

import numpy as np                 # Calculs numériques (matrices, moyennes, etc.)

import matplotlib                  # Bibliothèque graphique principale
matplotlib.use("TkAgg")            # Backend interactif : affichage dans une fenêtre Tk (hors Jupyter)
import matplotlib.pyplot as plt    # Interface pyplot pour créer/modifier la figure
import matplotlib.gridspec as gridspec  # Mise en page multi-panneaux avec ratios configurables
from matplotlib.lines import Line2D     # Éléments de légende personnalisés (lignes)
from matplotlib.patches import Patch    # Éléments de légende personnalisés (rectangles colorés)

from datetime import datetime, timedelta  # Gestion des heures simulées (étiquettes temporelles)
from collections import namedtuple        # Création du type Action (basal, bolus) comme tuple nommé

# ─── Imports SimGlucose ───────────────────────────────────────────────────────
from simglucose.simulation.env import T1DSimEnv          # Environnement de simulation DT1
from simglucose.simulation.scenario import Scenario      # Classe de base pour le scénario de repas
from simglucose.simulation.scenario import Action as MealAction  # Action repas (grammes de glucides)
from simglucose.patient.t1dpatient import T1DPatient     # Modèle patient UVa-Padova
from simglucose.sensor.cgm import CGMSensor              # Capteur de glucose en continu (CGM)
from simglucose.actuator.pump import InsulinPump         # Pompe à insuline virtuelle


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION PATIENT ADOLESCENT
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_NAME = "adolescent#001"   # Identifiant du patient dans la base SimGlucose

# Tableau des 5 repas de la journée simulée (cycle de 600 minutes = 10 heures)
REPAS_JOURNEE = [
    {"min_in_day":  60, "carbs": 45, "label": "Petit-dej\n45g"},   # Petit-déjeuner : 45 g de glucides à t=60 min
    {"min_in_day": 180, "carbs": 60, "label": "Dejeuner\n60g"},    # Déjeuner : 60 g à t=180 min
    {"min_in_day": 300, "carbs": 20, "label": "Gouter\n20g"},      # Goûter : 20 g à t=300 min
    {"min_in_day": 420, "carbs": 55, "label": "Diner\n55g"},       # Dîner : 55 g à t=420 min
    {"min_in_day": 540, "carbs": 15, "label": "Collation\n15g"},   # Collation : 15 g à t=540 min
]

ICR       = 15.0    # Insulin-to-Carb Ratio : 1 unité d'insuline pour 15 g de glucides
ISF       = 60.0    # Insulin Sensitivity Factor : 1 unité abaisse la glycémie de 60 mg/dL
CGM_CIBLE = 120.0   # Glycémie cible (mg/dL) utilisée pour le calcul du bolus de correction
BOLUS_MIN = 0.3     # Dose minimale de bolus autorisée (unités)
BOLUS_MAX = 10.0    # Dose maximale de bolus autorisée (unités), sécurité anti-surdosage


def calculer_bolus(carbs, cgm_val):
    """
    Calcule la dose de bolus à injecter avant un repas.
    Combine un bolus repas (basé sur les glucides) et un bolus de correction (si glycémie > 80).
    Retourne 0 si hypoglycémie détectée (CGM < 70 mg/dL).
    """
    if cgm_val < 70.0:                                          # Hypoglycémie : ne pas injecter
        return 0.0
    bolus_repas      = carbs / ICR                              # Dose insuline pour couvrir les glucides
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0  # Correction hyperglycémie
    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)  # Clamping et arrondi


MINUTES_PAR_JOUR = 600   # Durée d'un cycle de repas (5 repas sur 10 heures)
REPAS            = []    # Liste globale des repas (peuplée plus bas par build_repas_for_horizon)


def build_repas_for_horizon(sample_time, horizon_steps=10000):
    """
    Génère la liste complète des repas pour tout l'horizon de simulation.
    Calcule le nombre de jours nécessaires pour couvrir horizon_steps steps et
    répète le cycle REPAS_JOURNEE autant de fois que nécessaire.
    """
    repas     = []                                           # Liste de résultat
    total_min = horizon_steps * sample_time                  # Durée totale en minutes
    nb_jours  = int(total_min / MINUTES_PAR_JOUR) + 2       # +2 pour ne pas être court en fin de simulation
    for jour in range(nb_jours):                             # Boucle sur chaque jour simulé
        base_min = jour * MINUTES_PAR_JOUR                   # Minute de début du jour courant
        for r in REPAS_JOURNEE:                              # Boucle sur chaque repas du jour
            repas.append({
                "min"   : base_min + r["min_in_day"],        # Minute absolue du repas
                "carbs" : r["carbs"],                        # Grammes de glucides
                "bolus" : None,                              # Bolus calculé (rempli plus tard)
                "label" : r["label"],                        # Étiquette pour le graphe
            })
    return repas


# ─── Paramètres de la simulation de panne ────────────────────────────────────

PANNE_DUREE      = 32   # Nombre de steps pendant lesquels la pompe est en panne (ado : 32 × 3 min = 96 min)
MARCHE_DUREE     = 20   # Nombre de steps pendant lesquels la pompe fonctionne normalement (20 × 3 min = 60 min)
PANNE_START_STEP = 10   # Step à partir duquel le cycle panne/marche démarre
BASAL            = 0.040   # Débit basal de base pour l'adolescent (U/min)
PREBOLUS_STEPS   = 8       # Anticipation : le bolus est injecté 8 steps avant le repas
STEP_DELAY       = 0.0     # Délai en secondes entre deux steps (0.0 = exécution aussi rapide que possible)
SIM_DURATION_HOURS = 24    # Durée totale de la simulation en heures

Action = namedtuple("Action", ["basal", "bolus"])  # Type de données pour l'action envoyée à la pompe


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PARAMÈTRES LISA (adolescent)
# Logique : score pondéré sur 3 métriques, confirmé sur plusieurs steps consécutifs
# ─────────────────────────────────────────────────────────────────────────────

LISA_LW                 = 20        # Fenêtre longue (Long Window) : 20 steps = 60 min
LISA_SW                 = 3         # Fenêtre courte (Short Window) : 3 steps = 9 min
LISA_GFM_THRESHOLD      = 355.379392  # Seuil du GFM (Glucose Failure Metric cumulative)
LISA_IFM_THRESHOLD      = 0.320989    # Seuil de l'IFM (Insulin Failure Metric ratio)
LISA_SLOPE_THRESHOLD    = 0.942186    # Seuil de pente de la glycémie (mg/dL/min)
LISA_SCORE_MIN          = 3           # Score minimum pour déclencher une alarme brute (max = 4 pts)
LISA_PIE_ALPHA          = 0.85        # Facteur de décroissance exponentielle pour le PIE (Pharmacy of Insulin Effect)
LISA_CONFIRMATION_STEPS = 3           # Nombre de steps consécutifs avec score ≥ SCORE_MIN pour confirmer l'alerte


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PARAMÈTRES KALMAN (adolescent)
# ─────────────────────────────────────────────────────────────────────────────

KALMAN_WINDOW        = 20         # Fenêtre glissante pour la moyenne des résidus (20 steps = 60 min)
KALMAN_THRESHOLD     = 43.728771  # Seuil : alerte si la moyenne des résidus gelés dépasse cette valeur
KALMAN_CONFIRM_STEPS = 1          # Non utilisé dans cette version (conservé pour compatibilité)
KALMAN_SLOPE_MIN     = 1.000000   # Non utilisé dans cette version (conservé pour compatibilité)

# Matrices de bruit du filtre de Kalman (modèle d'état ordre 2 : position + vitesse)
Q_PROFILE = np.array([[1.0, 0.0],    # Bruit de processus Q :
                       [0.0, 0.10]])  #   [q_position, 0; 0, q_vitesse] — la vitesse varie peu
R_PROFILE = np.array([[25.0]])        # Bruit de mesure R : variance du capteur CGM (σ = 5 mg/dL)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PARAMÈTRES MACHINE À ÉTATS FUSION
# ─────────────────────────────────────────────────────────────────────────────

SEQ_WINDOW_STEPS = 40   # Fenêtre temporelle pour la suspicion séquentielle (40 steps ≈ 120 min)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — NOTIFICATIONS NTFY.SH
# Envoi d'alertes push vers des canaux ntfy.sh selon le niveau de détection
# ─────────────────────────────────────────────────────────────────────────────

NTFY_SUSPICION_URL       = "https://ntfy.sh/pompe_suspicion_detector"        # Canal suspicion simple
NTFY_SUSPICION_SEQ_URL   = "https://ntfy.sh/pompe_suspicion_sequentielle"    # Canal suspicion séquentielle
NTFY_PANNE_URL           = "https://ntfy.sh/pompe_panne_confirmee"           # Canal panne confirmée
NTFY_COOLDOWN_STEPS      = 10   # Nombre minimum de steps entre deux notifications du même type (anti-spam)

_ntfy_last_suspicion     = -999   # Step de la dernière notification suspicion simple (init très négatif)
_ntfy_last_suspicion_seq = -999   # Step de la dernière notification suspicion séquentielle
_ntfy_last_panne         = -999   # Step de la dernière notification panne confirmée


def _send_ntfy(url, title, message, priority="default", tags="rotating_light,warning"):
    """
    Envoie une notification HTTP POST vers ntfy.sh dans un thread séparé (daemon=False).
    Le thread daemon=False garantit que la notification est envoyée même si la simulation se termine.
    """
    def _do_send():
        try:
            data = message.encode("utf-8")                          # Encode le message en bytes UTF-8
            req  = urllib.request.Request(url, data=data, method="POST")  # Crée la requête POST
            req.add_header("Title",        title)                   # En-tête titre ntfy
            req.add_header("Priority",     priority)                # En-tête priorité (default/high/urgent)
            req.add_header("Tags",         tags)                    # En-tête tags (icônes emoji)
            req.add_header("Content-Type", "text/plain; charset=utf-8")  # Type de contenu
            with urllib.request.urlopen(req, timeout=8) as resp:    # Exécute la requête (timeout 8s)
                print(f"[NTFY] HTTP {resp.status} -> {url.split('/')[-1]}")  # Affiche le code HTTP reçu
        except Exception as e:
            print(f"[NTFY] Erreur envoi notification : {e}")        # Affiche l'erreur sans bloquer
    threading.Thread(target=_do_send, daemon=False).start()         # Lance le thread d'envoi


def notifier_suspicion(step):
    """Envoie une notification de suspicion simple si le cooldown est respecté."""
    global _ntfy_last_suspicion
    if step - _ntfy_last_suspicion >= NTFY_COOLDOWN_STEPS:          # Vérifie le cooldown anti-spam
        _ntfy_last_suspicion = step                                  # Met à jour le dernier step notifié
        _send_ntfy(NTFY_SUSPICION_URL,
                   "Suspicion pompe [ADO]",
                   "Anomalie detectee par un seul detecteur - surveillance renforcee.",
                   priority="high", tags="warning,mag")
        print(f"[NTFY] Notification SUSPICION envoyee (step {step})")


def notifier_suspicion_sequentielle(step, ecart_steps, sample_time):
    """Envoie une notification de suspicion séquentielle avec l'écart temporel entre les deux détecteurs."""
    global _ntfy_last_suspicion_seq
    if step - _ntfy_last_suspicion_seq >= NTFY_COOLDOWN_STEPS:      # Vérifie le cooldown
        _ntfy_last_suspicion_seq = step
        ecart_min = abs(ecart_steps) * sample_time                   # Convertit l'écart en minutes
        _send_ntfy(NTFY_SUSPICION_SEQ_URL,
                   "Suspicion elevee - alertes successives [ADO]",
                   f"Les deux detecteurs ont alerte successivement (ecart : {ecart_min:.0f} min).",
                   priority="high", tags="warning,rotating_light")
        print(f"[NTFY] Notification SUSPICION SEQUENTIELLE envoyee (step {step})")


def notifier_panne(step):
    """Envoie une notification de panne confirmée (priorité urgente) si le cooldown est respecté."""
    global _ntfy_last_panne
    if step - _ntfy_last_panne >= NTFY_COOLDOWN_STEPS:              # Vérifie le cooldown
        _ntfy_last_panne = step
        _send_ntfy(NTFY_PANNE_URL,
                   "PANNE CONFIRMEE [ADO]",
                   "Les deux detecteurs signalent une panne simultanee - verifiez la pompe.",
                   priority="urgent", tags="rotating_light,sos")    # Priorité maximale
        print(f"[NTFY] Notification PANNE CONFIRMEE envoyee (step {step})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — UTILITAIRES SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def pump_is_failed(step):
    """
    Retourne True si la pompe est en panne au step donné.
    Le cycle panne/marche démarre à PANNE_START_STEP :
      - panne pendant PANNE_DUREE steps
      - puis marche pendant MARCHE_DUREE steps
      - et ainsi de suite (modulo cyclique)
    """
    if step < PANNE_START_STEP:                                      # Avant le début du cycle : pompe OK
        return False
    return (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE) < PANNE_DUREE  # Position dans le cycle


class AdoScenario(Scenario):
    """
    Scénario de repas pour l'adolescent.
    Hérite de Scenario (SimGlucose) et fournit les actions repas au bon moment.
    """

    def __init__(self, start_time, repas, sample_time):
        self.start_time  = start_time    # Heure de départ de la simulation (datetime)
        self.repas       = repas         # Liste des repas (dicts avec "min", "carbs", etc.)
        self.sample_time = sample_time   # Durée d'un step en minutes
        # Dictionnaire step → glucides pour accès O(1) pendant la simulation
        self.meal_steps  = {int(r["min"] / sample_time): r["carbs"] for r in repas}

    def get_action(self, t):
        """
        Retourne l'action repas au temps t (datetime SimGlucose).
        Convertit t en step, et retourne les glucides du repas si ce step est prévu.
        """
        elapsed = (t - self.start_time).total_seconds() / 60   # Temps écoulé en minutes
        step    = int(round(elapsed / self.sample_time))        # Numéro de step correspondant
        return MealAction(meal=self.meal_steps.get(step, 0))    # 0 g si pas de repas ce step

    def reset(self):
        """Méthode requise par l'interface Scenario, ne fait rien ici."""
        pass

    def extend_if_needed(self, current_step, sample_time, horizon=500):
        """
        Étend dynamiquement le planning de repas si la simulation dépasse l'horizon calculé.
        Appelée à chaque step pour s'assurer qu'il y a toujours des repas planifiés
        au moins 'horizon' steps dans le futur.
        """
        max_step = max(self.meal_steps.keys()) if self.meal_steps else 0  # Dernier repas planifié
        if current_step + horizon > max_step:                              # Besoin d'extension
            last_min      = max_step * sample_time
            nb_jours_supp = int((current_step + horizon) * sample_time / MINUTES_PAR_JOUR) + 2
            for jour in range(nb_jours_supp):
                base_min = last_min + (jour + 1) * MINUTES_PAR_JOUR       # Base temporelle du nouveau jour
                for r in REPAS_JOURNEE:
                    new_min  = base_min + r["min_in_day"]
                    new_step = int(new_min / sample_time)
                    if new_step not in self.meal_steps:                    # Évite les doublons
                        self.meal_steps[new_step] = r["carbs"]
                        self.repas.append({
                            "min"  : new_min,
                            "carbs": r["carbs"],
                            "bolus": None,
                            "label": r["label"],
                            "step" : new_step,
                        })


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DÉTECTEUR LISA
# Logique : GFM cumulatif + IFM + pente → score pondéré → filtre de confirmation
# ─────────────────────────────────────────────────────────────────────────────

class LISADetector:
    """
    Détecteur LISA (Low Insulin Sensitivity Alert) adapté au profil adolescent.
    Calcule trois indicateurs de panne à chaque step :
      - GFM (Glucose Failure Metric) : montée cumulative de glycémie
      - IFM (Insulin Failure Metric) : baisse relative de l'insuline active
      - Pente : variation récente de la glycémie moyenne (fenêtre courte)
    Un score pondéré est calculé et confirmé sur plusieurs steps consécutifs.
    """

    def __init__(self, sample_time):
        self.sample_time     = sample_time  # Durée d'un step (minutes)
        self.cgm_history     = []           # Historique de toutes les mesures CGM
        self.pie_history     = []           # Historique du PIE (insuline active estimée)
        self._pie_current    = 0.0          # Valeur courante du PIE (filtre exponentiel)
        self._gfm_current    = 0.0          # Valeur cumulative actuelle du GFM
        self._sw_prev        = None         # Valeur précédente de la moyenne fenêtre courte (pour pente)
        self._or_consecutive = 0            # Compteur de steps consécutifs avec score ≥ SCORE_MIN
        self._confirmed      = False        # Alerte LISA confirmée (True si consec ≥ CONFIRMATION_STEPS)

    def update(self, cgm_val, insulin_dose):
        """
        Met à jour les indicateurs LISA avec la nouvelle mesure CGM et la dose d'insuline du step.
        Retourne : gfm, ifm, slope, confirmed, score_signal (bool brut), consecutive_count
        """
        dt = self.sample_time

        # ── PIE : insuline active estimée par filtre IIR (réponse impulsionnelle infinie) ──
        # PIE(t) = α × PIE(t-1) + dose(t)   avec α = LISA_PIE_ALPHA = 0.85
        # Plus α est proche de 1, plus la mémoire est longue
        self._pie_current = LISA_PIE_ALPHA * self._pie_current + insulin_dose
        self.pie_history.append(self._pie_current)   # Enregistre le PIE courant
        self.cgm_history.append(cgm_val)             # Enregistre la mesure CGM courante

        # Attente que l'historique soit suffisamment long pour calculer la fenêtre LW
        if len(self.cgm_history) < LISA_LW + 1:
            self._or_consecutive = 0
            self._confirmed      = False
            return 0.0, 0.0, 0.0, False, False, 0   # Pas encore assez de données

        # ── Calcul des moyennes fenêtre longue (LW) et courte (SW) ──
        cgm_arr    = np.array(self.cgm_history[-LISA_LW:])   # Tableau des LISA_LW derniers CGM
        pie_arr    = np.array(self.pie_history[-LISA_LW:])   # Tableau des LISA_LW derniers PIE
        glucose_lw = float(np.mean(cgm_arr))                 # Moyenne glycémie fenêtre longue
        glucose_sw = float(np.mean(self.cgm_history[-LISA_SW:]))  # Moyenne glycémie fenêtre courte
        pie_lw     = float(np.mean(pie_arr))                 # Moyenne PIE fenêtre longue
        pie_sw     = float(np.mean(self.pie_history[-LISA_SW:]))  # Moyenne PIE fenêtre courte

        # ── GFM : intégrale des hausses de glycémie (SW > LW) ──
        # Cumule les montées de glycémie sur le temps, se remet à 0 dès que SW ≤ LW
        diff = glucose_sw - glucose_lw           # Différence entre glycémie récente et moyenne longue
        if diff > 0:
            self._gfm_current = self._gfm_current + diff * dt  # Intégration numérique de la hausse
        else:
            self._gfm_current = 0.0              # Réinitialisation : la glycémie n'est plus en hausse
        gfm = self._gfm_current

        # ── IFM : baisse relative de l'insuline active (pie_lw > pie_sw) ──
        # Mesure la proportion de chute d'insuline par rapport à la référence longue
        ifm = max(0.0, (pie_lw - pie_sw) / pie_lw) if pie_lw > 1e-6 else 0.0

        # ── Pente de la glycémie (mg/dL/min) ──
        # Variation de la moyenne fenêtre courte entre deux steps consécutifs
        slope         = (glucose_sw - self._sw_prev) / dt if self._sw_prev is not None else 0.0
        self._sw_prev = glucose_sw   # Mémorise pour le prochain step

        # ── Score pondéré LISA ──
        # GFM compte double (poids 2) car indicateur principal ; IFM et Pente comptent pour 1 chacun
        lisa_score = (2 * int(gfm   > LISA_GFM_THRESHOLD) +   # +2 si GFM dépasse son seuil
                      1 * int(ifm   > LISA_IFM_THRESHOLD) +   # +1 si IFM dépasse son seuil
                      1 * int(slope > LISA_SLOPE_THRESHOLD))   # +1 si pente dépasse son seuil

        score_signal = lisa_score >= LISA_SCORE_MIN   # True si score brut ≥ 3 pts (sur 4 max)

        # ── Filtre de confirmation temporel ──
        # L'alerte n'est validée que si score_signal est True pendant LISA_CONFIRMATION_STEPS steps de suite
        if score_signal:
            self._or_consecutive += 1   # Incrémente le compteur de steps consécutifs
        else:
            self._or_consecutive = 0    # Réinitialise le compteur (rupture de continuité)
            self._confirmed      = False # L'alerte n'est plus confirmée

        if self._or_consecutive >= LISA_CONFIRMATION_STEPS:
            self._confirmed = True      # Confirmation officielle de l'alerte LISA

        return gfm, ifm, slope, self._confirmed, score_signal, self._or_consecutive

    def reset(self):
        """Réinitialise l'état interne de LISA (appelé si nécessaire entre les fenêtres)."""
        self._gfm_current    = 0.0
        self._or_consecutive = 0
        self._confirmed      = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — DÉTECTEUR KALMAN
# Filtre de Kalman d'ordre 2 avec mécanisme gel/dégel et fenêtre glissante
# ─────────────────────────────────────────────────────────────────────────────

class KalmanDetector:
    """
    Détecteur basé sur un filtre de Kalman à deux états : [glycémie, variation].

    Mécanisme gel/dégel :
    - Gel (freeze) : quand la glycémie CGM dépasse la prédiction (écart > 0) et monte,
      Kalman arrête de se corriger sur la mesure. Il accumule les résidus dans une fenêtre.
    - Dégel : automatique si la glycémie descend pendant N_DEGEL steps consécutifs.
      Kalman reprend sa correction normale et la fenêtre est vidée.

    Alerte : si la moyenne des résidus gelés dépasse KALMAN_THRESHOLD.
    """

    N_DEGEL = 5                  # Nombre de steps de descente consécutifs pour déclencher le dégel
    SLOPE_LOOKBACK_STEPS = 3     # Fenêtre de calcul de la pente CGM (non utilisée directement pour l'alerte)

    def __init__(self, first_cgm, sample_time):
        """Initialise le filtre Kalman avec la première mesure CGM."""
        self.dt = sample_time
        # Vecteur d'état initial : [glycémie, variation] avec variation = 0 (état stable supposé)
        self.x  = np.array([[first_cgm], [0.0]])
        self.P  = np.eye(2) * 10          # Matrice de covariance initiale (incertitude initiale = 10)
        # Matrice de transition d'état F : glycémie(t+1) = glycémie(t) + variation(t)*dt
        self.F  = np.array([[1, self.dt], [0, 1]])
        self.H  = np.array([[1, 0]])      # Matrice d'observation : on mesure seulement la glycémie
        self.Q  = Q_PROFILE               # Bruit de processus (défini globalement)
        self.R  = R_PROFILE               # Bruit de mesure (défini globalement)

        self.freeze              = False   # Indicateur de gel du filtre
        self._ecarts_fenetre     = []      # Fenêtre glissante des résidus gelés
        self._confirmed          = False   # Alerte Kalman active
        self._neg_slope_count    = 0       # Compteur de steps de descente CGM consécutifs
        self._cgm_history        = [float(first_cgm)]  # Historique CGM pour le calcul de pente
        self._prev_cgm           = float(first_cgm)    # Dernière mesure CGM (pour delta_cgm)

    def _slope_cgm(self):
        """Calcule la pente CGM sur SLOPE_LOOKBACK_STEPS steps (non utilisée pour l'alerte directe)."""
        if len(self._cgm_history) <= self.SLOPE_LOOKBACK_STEPS:
            return 0.0
        return (self._cgm_history[-1] - self._cgm_history[-1 - self.SLOPE_LOOKBACK_STEPS]) / (
            self.SLOPE_LOOKBACK_STEPS * self.dt
        )

    def update(self, cgm):
        """
        Effectue une mise à jour du filtre Kalman avec la nouvelle mesure CGM.
        Gère automatiquement le gel/dégel et calcule l'alerte.
        Retourne : prédiction libre, résidu, score moyen, alerte confirmée (bool)
        """
        cgm = float(cgm)
        self._cgm_history.append(cgm)          # Enregistre la mesure CGM

        # ── Étape de prédiction Kalman ──
        x_pred = self.F @ self.x               # Prédiction de l'état : x_k|k-1 = F × x_k-1
        P_pred = self.F @ self.P @ self.F.T + self.Q  # Prédiction de la covariance : P_k|k-1 = F×P×F' + Q
        z = np.array([[cgm]])                  # Vecteur mesure (scalaire dans un tableau 2D)

        # ── Prédiction libre et résidu ──
        p_libre  = float((self.H @ x_pred)[0, 0])     # Prédiction de la glycémie sans correction
        ecart    = float(cgm - p_libre)               # Écart entre mesure réelle et prédiction
        residual = float((z - self.H @ x_pred)[0, 0]) # Résidu = même chose (innovation)

        # ── Variation instantanée de la glycémie ──
        delta_cgm = cgm - self._prev_cgm        # Positif = montée, négatif = descente

        # ── Comptage des descentes consécutives pour le dégel ──
        if delta_cgm < 0:
            self._neg_slope_count += 1    # Descente : incrémente le compteur
        else:
            self._neg_slope_count = 0     # Montée ou stable : remet le compteur à 0

        # ── Dégel automatique si descente persistante ──
        just_degel = False
        if self.freeze and self._neg_slope_count >= self.N_DEGEL:
            self.freeze              = False   # Dégelé
            self._ecarts_fenetre     = []      # Vide la fenêtre glissante
            self._confirmed          = False   # L'alerte est annulée
            self._neg_slope_count    = 0       # Remet le compteur à 0
            just_degel               = True    # Signale que le dégel vient d'avoir lieu

        # ── Gel automatique si la glycémie monte et dépasse la prédiction ──
        # On ne gèle PAS si le CGM descend (delta_cgm < 0) pour éviter les faux positifs
        # On ne regèle PAS immédiatement après un dégel (just_degel = True)
        if (ecart > 0) and (not self.freeze) and (not just_degel) and (delta_cgm >= 0):
            self.freeze              = True    # Gel activé
            self._neg_slope_count    = 0       # Remet le compteur à 0

        if self.freeze:
            # ── Mode gelé : accumulation des résidus sans correction ──
            self._ecarts_fenetre.append(max(0.0, ecart))   # Ajoute le résidu positif à la fenêtre
            if len(self._ecarts_fenetre) > KALMAN_WINDOW:   # Maintient la fenêtre de taille KALMAN_WINDOW
                self._ecarts_fenetre.pop(0)
            score = float(np.mean(self._ecarts_fenetre)) if self._ecarts_fenetre else 0.0  # Score = moyenne

            # Mode gel : pas de correction par la mesure, prédiction libre seulement
            self.x = x_pred    # L'état prédit devient l'état courant (pas de correction)
            self.P = P_pred    # La covariance prédite devient la covariance courante
        else:
            # ── Mode normal : correction classique Kalman ──
            self._ecarts_fenetre = []   # Vide la fenêtre (Kalman suit normalement le CGM)
            score = 0.0

            S = self.H @ P_pred @ self.H.T + self.R           # Innovation covariance S = H×P×H' + R
            K = P_pred @ self.H.T @ np.linalg.inv(S)          # Gain de Kalman K = P×H' × S^-1
            self.x = x_pred + K @ (z - self.H @ x_pred)       # Correction de l'état : x_k = x_k|k-1 + K×innovation
            self.P = (np.eye(2) - K @ self.H) @ P_pred        # Mise à jour de la covariance : P_k = (I-K×H)×P_k|k-1

        # ── Décision d'alerte ──
        # Pas d'alerte si : Kalman vient de se dégeler, le CGM descend, ou Kalman n'est pas gelé
        if (not self.freeze) or just_degel or (delta_cgm < 0):
            self._confirmed = False                            # Alerte annulée
        else:
            self._confirmed = bool(score > KALMAN_THRESHOLD)  # Alerte si score moyen > seuil

        self._prev_cgm = cgm                          # Mémorise la mesure courante pour le prochain step
        return p_libre, residual, score, self._confirmed  # Retourne les 4 valeurs utiles

    def set_freeze(self, freeze):
        """
        Dans cette version (fusion passive), la fusion ne commande plus le gel/dégel.
        Cette méthode est conservée pour compatibilité mais ne fait rien.
        """
        pass

    def reset(self, cgm):
        """Réinitialise complètement le filtre Kalman à une nouvelle valeur de glycémie."""
        cgm = float(cgm)
        self.x              = np.array([[cgm], [0.0]])   # Réinit de l'état
        self.P              = np.eye(2) * 10             # Réinit de la covariance
        self.freeze         = False                      # Dégel forcé
        self._ecarts_fenetre = []                        # Vide la fenêtre
        self._confirmed     = False                      # Annule l'alerte
        self._neg_slope_count = 0
        self._cgm_history   = [cgm]
        self._prev_cgm      = cgm


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — MACHINE À ÉTATS FUSION
# Combine les alertes Kalman et LISA pour produire une décision de panne
# ─────────────────────────────────────────────────────────────────────────────

# Libellés des 5 cas de fusion (utilisés pour le comptage et l'affichage console)
CAS_KK_LL = "K_alerte  + L_alerte  → PANNE CONFIRMÉE"        # Les deux détecteurs alertent simultanément
CAS_KK_LS = "K_alerte  + L_silence → SUSPICION Kalman"       # Kalman seul en alerte
CAS_KS_LL = "K_silence + L_alerte  → SUSPICION LISA"         # LISA seul en alerte
CAS_KS_LS = "K_silence + L_silence → OK"                     # Aucun détecteur en alerte
CAS_SEQ   = "Alertes successives K→L ou L→K → SUSPICION SÉQUENTIELLE"  # Alertes dans la même fenêtre inter-bolus


class FusionStateMachine:
    """
    Machine à états pour la fusion passive des détecteurs Kalman et LISA.

    États possibles :
      - OK                    : aucun détecteur en alerte
      - SUSPICION             : un seul détecteur en alerte
      - SUSPICION_SEQUENTIELLE: les deux détecteurs ont alerté successivement dans la même fenêtre
      - PANNE                 : les deux détecteurs alertent au même step

    Règles :
      - La fusion lit seulement k_alert et l_alert. Elle ne commande jamais les détecteurs.
      - L'état PANNE n'est pas maintenu : si un détecteur s'arrête, l'état revient à SUSPICION ou OK.
      - La suspicion séquentielle est détectée via la mémoire de la dernière alerte simple.
    """

    def __init__(self):
        self.etat      = "OK"    # État courant de la fusion
        self.confiance = None    # Niveau de confiance ("HAUTE", "SEQUENTIELLE", None)

        # Compatibilité avec le code historique (ces commandes sont toujours False en v3)
        self.cmd_freeze = False   # La fusion ne gèle plus Kalman
        self.cmd_reset  = False   # La fusion ne réinitialise plus les détecteurs

        # Compteurs d'états (pour les statistiques finales)
        self.nb_panne         = 0   # Nombre de steps en état PANNE
        self.nb_suspicion     = 0   # Nombre de steps en état SUSPICION simple
        self.nb_suspicion_seq = 0   # Nombre de steps en état SUSPICION_SEQUENTIELLE
        self.nb_ok            = 0   # Nombre de steps en état OK
        self.nb_panne_haute   = 0   # Même chose que nb_panne (conservé pour compatibilité)

        # Compteurs par cas
        self.nb_cas = {
            CAS_KK_LL: 0,
            CAS_KK_LS: 0,
            CAS_KS_LL: 0,
            CAS_KS_LS: 0,
            CAS_SEQ:   0,
        }

        # Mémoire des premiers steps d'alerte dans la fenêtre courante
        self._k_first_alert_step = None   # Step de la 1ère alerte Kalman dans la fenêtre
        self._l_first_alert_step = None   # Step de la 1ère alerte LISA dans la fenêtre
        self.ecarts = []                  # Historique des écarts K→L (pour analyse)

        # Mémoire courte pour détecter la suspicion séquentielle inter-bolus
        self._last_single_detector = None   # Quel détecteur a alerté en dernier ("K" ou "L")
        self._last_single_step     = None   # À quel step
        self._last_single_window   = None   # Dans quelle fenêtre inter-bolus

    def _single_detector_courant(self, k_alert, l_alert):
        """Retourne "K", "L", ou None selon quel détecteur seul est en alerte."""
        if k_alert and not l_alert:
            return "K"     # Kalman seul
        if l_alert and not k_alert:
            return "L"     # LISA seul
        return None        # Aucun ou les deux

    def _est_sequentielle_interbolus(self, detector_now, step, bolus_window_start):
        """
        Vérifie si l'alerte courante constitue la deuxième moitié d'une alerte séquentielle.
        Conditions :
          1. Un seul détecteur alerte maintenant
          2. Un seul détecteur différent a alerté récemment
          3. Les deux alertes sont dans la même fenêtre inter-bolus
          4. L'écart temporel est ≤ SEQ_WINDOW_STEPS
        """
        if detector_now not in ("K", "L"):       return False   # Pas d'alerte simple maintenant
        if self._last_single_detector not in ("K", "L"): return False  # Pas de mémoire
        if detector_now == self._last_single_detector:   return False  # Même détecteur = pas séquentiel
        if bolus_window_start is None or self._last_single_window is None: return False
        if bolus_window_start != self._last_single_window: return False  # Fenêtres inter-bolus différentes
        return 0 < (step - self._last_single_step) <= SEQ_WINDOW_STEPS  # Dans la fenêtre temporelle

    def update(self, k_alert, l_alert, step, cgm_val, sample_time, bolus_window_start=None):
        """
        Met à jour l'état de la fusion à chaque step.
        Paramètres :
          - k_alert, l_alert : booléens des alertes Kalman et LISA
          - step             : numéro du step courant
          - cgm_val          : glycémie mesurée (pour information)
          - sample_time      : durée d'un step
          - bolus_window_start : début de la fenêtre inter-bolus courante (ou None)
        Retourne : (etat_fusion, confiance, cas)
        """
        # Fusion passive : jamais de commande vers les détecteurs
        self.cmd_freeze = False
        self.cmd_reset  = False

        detector_now = self._single_detector_courant(k_alert, l_alert)  # Identifie le détecteur actif

        if k_alert and l_alert:
            # ── CAS 1 : PANNE confirmée — les deux détecteurs alertent simultanément ──
            cas            = CAS_KK_LL
            self.etat      = "PANNE"
            self.confiance = "HAUTE"

            # Mémorise le premier step d'alerte de chaque détecteur dans cette séquence
            if self._k_first_alert_step is None:
                self._k_first_alert_step = step
            if self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._enregistrer_ecart(step, "HAUTE")  # Enregistre l'écart K→L

            # Remet à zéro la mémoire séquentielle (la panne simultanée prend le dessus)
            self._last_single_detector = None
            self._last_single_step     = None
            self._last_single_window   = None

        elif detector_now and self._est_sequentielle_interbolus(detector_now, step, bolus_window_start):
            # ── CAS 2 : SUSPICION SÉQUENTIELLE — un détecteur a alerté après l'autre dans la même fenêtre ──
            cas            = CAS_SEQ
            self.etat      = "SUSPICION_SEQUENTIELLE"
            self.confiance = "SEQUENTIELLE"

            # Mémorise le step d'alerte du détecteur qui vient d'alerter
            if detector_now == "K" and self._k_first_alert_step is None:
                self._k_first_alert_step = step
            if detector_now == "L" and self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._enregistrer_ecart(step, "SEQUENTIELLE")

            # Met à jour la mémoire séquentielle sans la réinitialiser
            self._last_single_detector = detector_now
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        elif k_alert and not l_alert:
            # ── CAS 3 : SUSPICION Kalman — Kalman seul en alerte ──
            cas            = CAS_KK_LS
            self.etat      = "SUSPICION"
            self.confiance = None

            if self._k_first_alert_step is None:
                self._k_first_alert_step = step

            # Mémorise l'alerte Kalman pour éventuellement former une paire séquentielle
            self._last_single_detector = "K"
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        elif l_alert and not k_alert:
            # ── CAS 4 : SUSPICION LISA — LISA seul en alerte ──
            cas            = CAS_KS_LL
            self.etat      = "SUSPICION"
            self.confiance = None

            if self._l_first_alert_step is None:
                self._l_first_alert_step = step

            # Mémorise l'alerte LISA pour éventuellement former une paire séquentielle
            self._last_single_detector = "L"
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        else:
            # ── CAS 5 : OK — aucun détecteur en alerte ──
            cas            = CAS_KS_LS
            self.etat      = "OK"
            self.confiance = None
            self._reset_first_alerts()   # Réinitialise les premiers steps d'alerte
            # Note : on ne réinitialise PAS _last_single_* ici
            # pour permettre la détection séquentielle même après un court silence

        self.nb_cas[cas] += 1   # Incrémente le compteur du cas courant

        # Mise à jour des compteurs d'états
        if self.etat == "PANNE":
            self.nb_panne      += 1
            self.nb_panne_haute += 1
        elif self.etat == "SUSPICION_SEQUENTIELLE":
            self.nb_suspicion_seq += 1
        elif self.etat == "SUSPICION":
            self.nb_suspicion += 1
        else:
            self.nb_ok += 1

        return self.etat, self.confiance, cas

    def _enregistrer_ecart(self, step, confiance):
        """Calcule et enregistre l'écart temporel entre les premiers steps d'alerte K et L."""
        if (self._k_first_alert_step is not None and
                self._l_first_alert_step is not None):
            ecart = self._l_first_alert_step - self._k_first_alert_step  # Positif = L après K
            self.ecarts.append((step, ecart, confiance))
        elif self._k_first_alert_step is not None:
            self.ecarts.append((step, None, confiance))   # LISA n'a pas encore alerté
        elif self._l_first_alert_step is not None:
            self.ecarts.append((step, None, confiance))   # Kalman n'a pas encore alerté

    def _reset_first_alerts(self):
        """Réinitialise les mémoires de premiers steps d'alerte (nouvelle fenêtre)."""
        self._k_first_alert_step = None
        self._l_first_alert_step = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — GESTIONNAIRE DE MÉTRIQUES VP/FP/FN (par fenêtre inter-bolus)
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Calcule les métriques VP / FP / FN par fenêtre inter-bolus.

    Définition des fenêtres :
      - Une fenêtre commence à chaque bolus prévu et se termine au bolus suivant.
      - La vérité terrain de la fenêtre est déterminée par le premier bolus :
        si la pompe est HS à ce step = panne réelle.
      - Une pompe HS sans bolus prévu ne compte pas comme panne détectable.

    Définitions VP / FP / FN :
      - VP : bolus manqué ET détection dans la fenêtre → correct
      - FN : bolus manqué ET aucune détection → raté
      - FP : bolus injecté ET détection dans la fenêtre → fausse alarme

    Mode fusion : détection = les deux détecteurs K ET L ont alerté dans la même fenêtre.
    Mode single : détection = le détecteur a alerté au moins une fois dans la fenêtre.
    """

    def __init__(self, name, steps_24h, mode="single"):
        self.name      = name       # Nom du détecteur ("KALMAN", "LISA", "FUSION")
        self.steps_24h = steps_24h  # Nombre de steps pour 24h (pour affichage périodique)
        self.mode      = mode       # "single" ou "fusion"

        self.vp = 0       # Vrais Positifs cumulés
        self.fp = 0       # Faux Positifs cumulés
        self.fn = 0       # Faux Négatifs cumulés
        self.archives = []  # Historique des métriques périodiques (toutes les 24h)

        self._bolus_steps_sorted = []   # Liste triée des steps de bolus prévus
        self._win_start  = None         # Début de la fenêtre inter-bolus courante
        self._win_end    = None         # Fin de la fenêtre inter-bolus courante (borne exclue)
        self._win_manque = False        # La panne réelle est-elle présente dans cette fenêtre ?

        # Pour détecteur simple (mode "single")
        self._win_alerted = False       # Au moins une alerte dans la fenêtre ?

        # Pour la fusion (mode "fusion")
        self._win_k_alerted    = False  # Kalman a-t-il alerté dans la fenêtre ?
        self._win_l_alerted    = False  # LISA a-t-il alerté dans la fenêtre ?
        self._win_simultaneous = False  # Alerte simultanée dans la fenêtre ?
        self._win_first_k      = None   # Step de la première alerte Kalman dans la fenêtre
        self._win_first_l      = None   # Step de la première alerte LISA dans la fenêtre

    def register_bolus_steps(self, bolus_steps_dict):
        """Met à jour la liste triée des bolus prévus (appelée à chaque step pour rester à jour)."""
        self._bolus_steps_sorted = sorted(bolus_steps_dict.keys())

    def _find_window(self, step):
        """
        Trouve la fenêtre inter-bolus contenant le step courant via recherche binaire.
        Retourne (win_start, win_end) ou (None, None) si hors plage.
        """
        bs = self._bolus_steps_sorted
        if not bs or step < bs[0]:
            return None, None           # Pas encore dans une fenêtre bolus

        lo, hi = 0, len(bs) - 1
        idx = 0
        while lo <= hi:                 # Recherche binaire du dernier bolus ≤ step
            mid = (lo + hi) // 2
            if bs[mid] <= step:
                idx = mid
                lo  = mid + 1
            else:
                hi  = mid - 1

        win_start = bs[idx]                                         # Début de fenêtre = bolus courant
        win_end   = bs[idx + 1] if idx + 1 < len(bs) else None     # Fin = bolus suivant (ou None si dernier)
        return win_start, win_end

    def _open_new_window(self, win_start, win_end, bolus_manque_ce_step):
        """
        Ouvre une nouvelle fenêtre inter-bolus.
        La vérité terrain (pompe HS ?) est fixée à l'entrée de la fenêtre.
        """
        self._win_start  = win_start
        self._win_end    = win_end
        # La panne est définie par le bolus de début de fenêtre, pas par l'état général de la pompe
        self._win_manque = bool(bolus_manque_ce_step)   # True si bolus manqué à win_start

        # Réinitialise les indicateurs d'alerte pour cette nouvelle fenêtre
        self._win_alerted      = False
        self._win_k_alerted    = False
        self._win_l_alerted    = False
        self._win_simultaneous = False
        self._win_first_k      = None
        self._win_first_l      = None

    def update_single(self, step, alert, bolus_manque_ce_step):
        """Met à jour les métriques pour un détecteur simple (Kalman seul ou LISA seul)."""
        win_start, win_end = self._find_window(step)
        if win_start is None:
            return    # Pas encore de fenêtre définie

        if win_start != self._win_start:         # Changement de fenêtre inter-bolus
            if self._win_start is not None:
                self._close_window()             # Clôture la fenêtre précédente → calcule VP/FP/FN
            self._open_new_window(win_start, win_end, bolus_manque_ce_step)

        if alert:
            self._win_alerted = True             # Mémorise qu'une alerte s'est produite dans cette fenêtre

        # Affichage et archivage périodique toutes les 24h
        if (step > 0) and (step % self.steps_24h == 0):
            self._print_and_archive(step)
            self._reset_counters_only()

    def update_fusion(self, step, k_alert, l_alert, bolus_manque_ce_step):
        """Met à jour les métriques pour la fusion (requiert K ET L dans la même fenêtre)."""
        win_start, win_end = self._find_window(step)
        if win_start is None:
            return

        if win_start != self._win_start:
            if self._win_start is not None:
                self._close_window()
            self._open_new_window(win_start, win_end, bolus_manque_ce_step)

        if k_alert:
            self._win_k_alerted = True               # Kalman a alerté dans cette fenêtre
            if self._win_first_k is None:
                self._win_first_k = step             # Mémorise le premier step d'alerte Kalman

        if l_alert:
            self._win_l_alerted = True               # LISA a alerté dans cette fenêtre
            if self._win_first_l is None:
                self._win_first_l = step             # Mémorise le premier step d'alerte LISA

        if k_alert and l_alert:
            self._win_simultaneous = True            # Alerte simultanée détectée

        # Affichage périodique
        if (step > 0) and (step % self.steps_24h == 0):
            self._print_and_archive(step)
            self._reset_counters_only()

    def update(self, step, alert, pump_failed=None, bolus_manque_ce_step=False):
        """Méthode de compatibilité — délègue à update_single."""
        self.update_single(step, alert, bolus_manque_ce_step)

    def _window_detected(self):
        """
        Retourne True si la fenêtre est considérée comme détectée.
        Mode fusion : les deux détecteurs doivent avoir alerté (simultané ou séquentiel).
        Mode single : au moins une alerte suffit.
        """
        if self.mode == "fusion":
            return self._win_k_alerted and self._win_l_alerted   # ET logique
        return self._win_alerted

    def _close_window(self):
        """
        Clôture la fenêtre inter-bolus courante et met à jour VP / FP / FN.
        Règle : une suspicion simple en fusion (1 seul détecteur) ne compte ni VP ni FP.
        """
        detected = self._window_detected()

        if self._win_manque:        # Panne réelle dans cette fenêtre
            if detected:
                self.vp += 1        # VP : panne détectée correctement
            else:
                self.fn += 1        # FN : panne non détectée
        else:                       # Pas de panne dans cette fenêtre
            if detected:
                self.fp += 1        # FP : fausse alarme (alerte sans panne)
                # En fusion : FP seulement si les DEUX détecteurs ont alerté sans raison

    def _scores(self):
        """Calcule précision, rappel et score F1 à partir des compteurs courants."""
        precision = self.vp / max(1, self.vp + self.fp) * 100   # Évite la division par zéro
        rappel    = self.vp / max(1, self.vp + self.fn) * 100
        f1        = 2 * precision * rappel / max(0.01, precision + rappel)
        return precision, rappel, f1

    def _print_and_archive(self, step):
        """Affiche les métriques périodiques (toutes les 24h) et les archive."""
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
            "vp": self.vp, "fp": self.fp, "fn": self.fn,
            "precision": precision, "rappel": rappel, "f1": f1,
        })

    def _reset_counters_only(self):
        """
        Remet les compteurs VP/FP/FN à 0 sans fermer la fenêtre inter-bolus courante.
        (La fenêtre peut chevaucher la frontière d'affichage 24h.)
        """
        self.vp = 0
        self.fp = 0
        self.fn = 0

    def resume_final(self, step):
        """
        Calcule et affiche les métriques finales à l'arrêt de la simulation.
        Ferme la fenêtre courante seulement si elle est complète (bolus suivant atteint).
        """
        if (self._win_start is not None and
                self._win_end   is not None and
                step >= self._win_end):
            self._close_window()   # Ferme la dernière fenêtre si elle est terminée
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CALCUL FINAL DES MÉTRIQUES ROBUSTES SUR 24H EXACTES
# ─────────────────────────────────────────────────────────────────────────────

def _calc_metrics_interbolus_final(name, bolus_steps_dict, missed_bolus_history,
                                   k_alert_history=None, l_alert_history=None,
                                   single_alert_history=None, n_steps=0):
    """
    Recalcul final et robuste des métriques VP/FP/FN sur les 24h complètes.

    Contrairement au MetricsTracker en temps réel, cette fonction parcourt
    toutes les fenêtres inter-bolus complètes a posteriori depuis les historiques.

    La dernière fenêtre incomplète (bolus de fin non encore atteint) est ignorée.
    """
    bolus_steps = sorted(s for s in bolus_steps_dict.keys() if s < n_steps)  # Bolus dans la simulation

    vp = fp = fn = 0    # Compteurs
    details = []        # Détails par fenêtre pour débogage

    for i in range(len(bolus_steps) - 1):
        bs = bolus_steps[i]      # Début de fenêtre (step du bolus courant)
        be = bolus_steps[i + 1]  # Fin de fenêtre (step du bolus suivant, borne exclue)

        if bs >= n_steps or be > n_steps:
            continue    # Ignore les fenêtres hors de la durée de simulation

        # Vérité terrain : le bolus a-t-il été manqué à ce step ?
        missed = bool(missed_bolus_history[bs]) if bs < len(missed_bolus_history) else False

        if single_alert_history is not None:
            # Mode détecteur simple : alerte = au moins un True dans [bs, be[
            detected = bool(np.any(single_alert_history[bs:be]))
            k_seen   = l_seen = None
        else:
            # Mode fusion : alerte = K ET L ont tous les deux alerté dans [bs, be[
            k_seen   = bool(np.any(k_alert_history[bs:be]))
            l_seen   = bool(np.any(l_alert_history[bs:be]))
            detected = k_seen and l_seen   # ET logique

        if missed:
            if detected:
                vp  += 1; cls = "VP"    # Panne détectée
            else:
                fn  += 1; cls = "FN"    # Panne manquée
        else:
            if detected:
                fp  += 1; cls = "FP"    # Fausse alarme
            else:
                cls = "TN"              # Vrai négatif (pas compté)

        details.append((bs, be, missed, detected, cls, k_seen, l_seen))

    # Calcul des scores
    precision = vp / max(1, vp + fp) * 100
    rappel    = vp / max(1, vp + fn) * 100
    f1        = 2 * precision * rappel / max(0.01, precision + rappel)

    # Affichage du tableau de résultats
    sep = "─" * 72
    print()
    print("═" * 72)
    print(f"  MÉTRIQUES FINALES ROBUSTES [{name}] — 24 h exactes")
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
        "name": name, "vp": vp, "fp": fp, "fn": fn,
        "precision": precision, "rappel": rappel, "f1": f1,
        "details": details,
    }


def afficher_resume_metriques_24h(bolus_steps_dict, missed_bolus_history,
                                   k_alert_history, l_alert_history, n_steps):
    """
    Affiche le résumé final des métriques pour les 3 détecteurs (Kalman, LISA, Fusion).
    Appelée à la fin de la simulation (KeyboardInterrupt ou fin automatique à 24h).
    """
    print(f"\n{'='*72}")
    print("  MÉTRIQUES FINALES — RÉSUMÉ DES 3 DÉTECTEURS")
    print("  Logique stricte : 1 bolus manqué = 1 fenêtre à détecter avant le bolus suivant")
    print(f"  Durée évaluée : {n_steps * sample_time / 60:.1f} h")
    print(f"{'='*72}")

    # Calcul pour Kalman seul
    r_k = _calc_metrics_interbolus_final(
        "KALMAN", bolus_steps_dict, missed_bolus_history,
        single_alert_history=np.array(k_alert_history, dtype=bool),
        n_steps=n_steps,
    )

    # Calcul pour LISA seul
    r_l = _calc_metrics_interbolus_final(
        "LISA", bolus_steps_dict, missed_bolus_history,
        single_alert_history=np.array(l_alert_history, dtype=bool),
        n_steps=n_steps,
    )

    # Calcul pour la Fusion (K ET L requis)
    r_f = _calc_metrics_interbolus_final(
        "FUSION", bolus_steps_dict, missed_bolus_history,
        k_alert_history=np.array(k_alert_history, dtype=bool),
        l_alert_history=np.array(l_alert_history, dtype=bool),
        n_steps=n_steps,
    )

    return r_k, r_l, r_f


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — INITIALISATION PATIENT & ENVIRONNEMENT
# ─────────────────────────────────────────────────────────────────────────────

patient    = T1DPatient.withName(PATIENT_NAME)          # Charge le patient adolescent depuis SimGlucose
sensor     = CGMSensor.withName("Dexcom", seed=1)       # Initialise le capteur Dexcom (seed fixe pour reproductibilité)
pump       = InsulinPump.withName("Insulet")            # Initialise la pompe Insulet
start_time = datetime.now()                             # Heure de départ (utilisée pour les étiquettes temporelles)

# Créer un environnement temporaire juste pour récupérer sample_time
env_tmp     = T1DSimEnv(patient=patient, sensor=sensor, pump=pump,
                        scenario=AdoScenario(start_time, [], 3))
sample_time = env_tmp.sample_time   # Récupère la durée d'un step en minutes (généralement 3 min)
del env_tmp                         # Supprime l'environnement temporaire

# Génération complète des repas pour tout l'horizon
REPAS    = build_repas_for_horizon(sample_time, horizon_steps=10000)

# Réinitialisation des objets SimGlucose pour la vraie simulation
patient  = T1DPatient.withName(PATIENT_NAME)
sensor   = CGMSensor.withName("Dexcom", seed=1)
pump     = InsulinPump.withName("Insulet")
scenario = AdoScenario(start_time, REPAS, sample_time)         # Scénario avec tous les repas
env      = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

# Association de chaque repas à son numéro de step
for r in REPAS:
    r["step"] = int(r["min"] / sample_time)   # Calcule le step correspondant à chaque repas

# Dictionnaire des steps de bolus : {step_bolus: carbs_associés}
# PREBOLUS_STEPS steps avant le repas pour tenir compte du délai d'action de l'insuline
BOLUS_CARBS_STEPS = {max(0, r["step"] - PREBOLUS_STEPS): r["carbs"] for r in REPAS}
BOLUS_STEPS       = BOLUS_CARBS_STEPS   # Alias pour l'affichage graphique


def get_inter_bolus_window_start(step, bolus_steps_dict):
    """
    Retourne le dernier step de bolus prévu ≤ step (début de la fenêtre inter-bolus courante).
    Utilise une recherche binaire pour l'efficacité.
    """
    bs = sorted(bolus_steps_dict.keys())
    if not bs or step < bs[0]:
        return None   # Pas encore dans une fenêtre
    lo, hi, idx = 0, len(bs) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if bs[mid] <= step: idx = mid; lo = mid + 1
        else:               hi  = mid - 1
    return bs[idx]


def update_bolus_steps(current_step, scenario, sample_time):
    """
    Étend la liste des bolus si nécessaire et met à jour BOLUS_CARBS_STEPS.
    Assure qu'il y a toujours des bolus planifiés au moins 500 steps en avance.
    """
    scenario.extend_if_needed(current_step, sample_time, horizon=500)
    for r in scenario.repas:
        s_r = r.get("step")
        if s_r is None: continue
        bolus_step = max(0, s_r - PREBOLUS_STEPS)            # Step du bolus (avec pré-bolus)
        if bolus_step not in BOLUS_CARBS_STEPS:
            BOLUS_CARBS_STEPS[bolus_step] = r["carbs"]       # Ajoute le nouveau bolus


# Calcul des constantes de durée pour la simulation
STEPS_24H     = int((24 * 60) / sample_time)           # Nombre de steps pour 24 heures
MAX_SIM_STEPS = int((SIM_DURATION_HOURS * 60) / sample_time)  # Nombre total de steps à simuler

# Affichage de la configuration initiale dans la console
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
print(f"  Kalman seuil     : {KALMAN_THRESHOLD}  |  fenêtre : {KALMAN_WINDOW} steps ({KALMAN_WINDOW*sample_time:.0f} min)")
print("  Kalman décision  : directe si moyenne résidus > seuil | dégel après 3 steps descendants")
print(f"  Cycle pannes     : {PANNE_DUREE} steps HS ({PANNE_DUREE*sample_time:.0f} min) → "
      f"{MARCHE_DUREE} steps OK ({MARCHE_DUREE*sample_time:.0f} min)")
print(f"  Repas            : {len(REPAS_JOURNEE)} repas/jour, cycle {MINUTES_PAR_JOUR} min")
print(f"  Bolus            : ICR={ICR}g/U | ISF={ISF}mg/dL/U")
print(f"  Durée évaluée    : {SIM_DURATION_HOURS} h exactes ({MAX_SIM_STEPS} steps)")
print(f"  Arrêt            : automatique à 24h ou Ctrl+C")
print(f"{'='*72}\n")

# En-tête du tableau de suivi en console
print(f"{'Step':>5} {'Heure':>6} {'CGM':>7} "
      f"{'K_alr':>7} {'L_alr':>7} {'Score':>8} {'Consec':>7} "
      f"{'GFM':>10} {'IFM':>7} {'Pente':>7} "
      f"{'Fusion':>22} {'Confiance':>14} {'Détecteur':>15} "
      f"{'Cas':>42} {'Réel':>8}")
print("─" * 170)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — CONFIGURATION DU GRAPHE (4 panneaux)
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("default")   # Style matplotlib par défaut (fond blanc)

# Palette de couleurs — vert-bleu pour distinguer l'adolescent (enfant=bleu)
C_BG        = "white"
C_GRID      = "#cccccc"    # Grille grise claire
C_SPINE     = "#aaaaaa"    # Bordures des axes
C_TICK      = "#333333"    # Couleur des étiquettes
C_CGM       = "#005f5f"    # Courbe CGM : vert-bleu profond
C_PRED      = "#b35900"    # Courbe prédiction Kalman : orange foncé
C_KSCORE    = "#8b0000"    # Score Kalman : rouge sombre
C_GFM       = "#4b0082"    # GFM LISA : indigo
C_SLOPE     = "#005f5f"    # Pente LISA : vert-bleu (même que CGM)
C_IFM       = "#7a3500"    # IFM LISA : brun
C_OR_BRUT   = "#aaaaaa"    # Score brut LISA : gris
C_SEUIL_K   = "#cc0000"    # Ligne de seuil Kalman : rouge
C_SEUIL_GFM = "#4b0082"    # Ligne de seuil GFM : indigo
C_SEUIL_SLP = "#005f5f"    # Ligne de seuil pente : vert-bleu
C_SEUIL_IFM = "#7a3500"    # Ligne de seuil IFM : brun
C_MEAL      = "#cc5500"    # Marqueurs repas : orange vif
C_BOLUS_OK  = "#003f8a"    # Marqueurs bolus injectés : bleu marine
C_BOLUS_FAIL= "#cc0000"    # Marqueurs bolus manqués : rouge
C_HAUTE     = "#cc0000"    # Alertes panne confirmée : rouge
C_SUSPICION = "#005f5f"    # Alertes suspicion : vert-bleu
C_SEQ       = "#b87700"    # Alertes suspicion séquentielle : ambre
C_BAND_FAIL = "#ffcccc"    # Bandes de fond pompe HS : rouge pâle
C_BAND_OK   = "#e6f7ee"    # Bandes de fond pompe OK : vert pâle
C_ZONE_CIBLE= "#e8f4fd"    # Zone glycémique cible [70-180] : bleu très pâle

fig = plt.figure(figsize=(16, 10))   # Figure de 16×10 pouces
fig.patch.set_facecolor(C_BG)         # Fond de la figure en blanc

# GridSpec : 4 panneaux empilés verticalement avec ratios [3:1:1:1]
# Panneau CGM plus grand (ratio 3) car le plus informatif
gs = gridspec.GridSpec(4, 1, figure=fig,
                       height_ratios=[3, 1, 1, 1], hspace=0.08)

ax_cgm = fig.add_subplot(gs[0])                   # Panneau 1 : glycémie + alertes
ax_k   = fig.add_subplot(gs[1], sharex=ax_cgm)    # Panneau 2 : score Kalman (même axe X)
ax_l   = fig.add_subplot(gs[2], sharex=ax_cgm)    # Panneau 3 : indicateurs LISA
ax_fus = fig.add_subplot(gs[3], sharex=ax_cgm)    # Panneau 4 : décision de fusion
AXES   = [ax_cgm, ax_k, ax_l, ax_fus]             # Liste des 4 axes pour les opérations communes

# Style commun à tous les axes
for ax in AXES:
    ax.set_facecolor(C_BG)                                    # Fond blanc
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)    # Grille grise légère
    ax.tick_params(colors=C_TICK, labelsize=7)                # Couleur et taille des graduations
    for s in ["top", "right"]:   ax.spines[s].set_visible(False)   # Cache les bordures top/right
    for s in ["bottom", "left"]: ax.spines[s].set_color(C_SPINE)   # Colorie les bordures bottom/left

# ── Panneau CGM — lignes et marqueurs ──
line_cgm,  = ax_cgm.plot([], [], color=C_CGM,  lw=2.5, zorder=3)              # Courbe CGM principale
line_pred, = ax_cgm.plot([], [], color=C_PRED, ls="--", lw=1.5, zorder=3, alpha=0.85)  # Prédiction Kalman (tirets)
sc_meals       = ax_cgm.scatter([], [], marker="^", s=160, color=C_MEAL,       zorder=6)  # Triangles repas
sc_bolus_ok    = ax_cgm.scatter([], [], marker="o", s=120, color=C_BOLUS_OK,   zorder=6)  # Cercles bolus OK
sc_bolus_fail  = ax_cgm.scatter([], [], marker="x", s=160, color=C_BOLUS_FAIL, zorder=6, lw=2.5)  # Croix bolus manqués
sc_panne_haute = ax_cgm.scatter([], [], marker="D", s=180, color=C_HAUTE,      zorder=9, lw=1.5)  # Losanges PANNE
sc_suspicion   = ax_cgm.scatter([], [], marker="o", s=80,  color=C_SUSPICION,  zorder=8,
                                facecolors="none", lw=2.0)   # Cercles vides : suspicion simple
sc_susp_seq    = ax_cgm.scatter([], [], marker="s", s=100, color=C_SEQ,        zorder=8,
                                facecolors="none", lw=2.0)   # Carrés vides : suspicion séquentielle

# Lignes horizontales de référence glycémique
ax_cgm.axhline(70,  color=C_BOLUS_OK,   ls="--", lw=1.2, alpha=0.7)  # Seuil hypoglycémie 70 mg/dL
ax_cgm.axhline(180, color=C_BOLUS_FAIL, ls="--", lw=1.2, alpha=0.7)  # Seuil hyperglycémie 180 mg/dL
ax_cgm.axhspan(70, 180, alpha=0.08, color=C_ZONE_CIBLE)               # Zone cible [70-180] en bleu pâle
ax_cgm.set_ylim(40, 450)                                               # Limites Y de l'axe CGM
ax_cgm.set_ylabel("Glycémie (mg/dL)", color=C_TICK, fontsize=8)

# Légende du panneau CGM
legend_elems = [
    Line2D([0],[0], color=C_CGM,   lw=2.5,                            label="CGM"),
    Line2D([0],[0], color=C_PRED,  lw=1.5, ls="--",                   label="Prédiction Kalman"),
    Patch(facecolor=C_BAND_FAIL, alpha=0.5,                            label="Pompe HS (réel)"),
    Patch(facecolor=C_BAND_OK,   alpha=0.5,                            label="Pompe OK (réel)"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor=C_MEAL,     ms=8, ls="None", label="Repas"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BOLUS_OK, ms=7, ls="None", label="Bolus injecté"),
    Line2D([0],[0], marker="x", color=C_BOLUS_FAIL,                    ms=8, ls="None", label="Bolus manqué"),
    Line2D([0],[0], marker="D", color=C_HAUTE,                         ms=8, ls="None", label="PANNE CONFIRMÉE (K+L)"),
    Line2D([0],[0], marker="o", color=C_SUSPICION,                     ms=8, ls="None",
           markerfacecolor="none",                                             label="SUSPICION (1 détecteur)"),
    Line2D([0],[0], marker="s", color=C_SEQ,                           ms=8, ls="None",
           markerfacecolor="none",                                             label="SUSPICION SÉQUENTIELLE"),
]
ax_cgm.legend(handles=legend_elems, loc="upper left",
              fontsize=7, facecolor="white", edgecolor=C_SPINE, ncol=3)

# ── Panneau Kalman ──
line_kscore, = ax_k.plot([], [], color=C_KSCORE, lw=2.0)                    # Courbe du score Kalman
ax_k.axhline(KALMAN_THRESHOLD, color=C_SEUIL_K, ls="--", lw=1.2, alpha=0.85)  # Ligne de seuil
ax_k.set_ylim(0, max(80, KALMAN_THRESHOLD * 1.8))                           # Y limité à 1.8× le seuil
ax_k.set_ylabel("Score\nKalman", color=C_TICK, fontsize=7)

# ── Panneau LISA ──
line_gfm,    = ax_l.plot([], [], color=C_GFM,    lw=2.0,       label="GFM/100")   # GFM divisé par 100 (mise à l'échelle)
line_ifm,    = ax_l.plot([], [], color=C_IFM,    lw=1.5, ls="-.", label="IFM")    # IFM (tiret-point)
line_slope,  = ax_l.plot([], [], color=C_SLOPE,   lw=1.5,       label="Pente")    # Pente glycémique
line_consec, = ax_l.plot([], [], color=C_OR_BRUT, lw=1.0, ls=":",
                          label=f"Consécutifs/{LISA_CONFIRMATION_STEPS}")          # Compteur confirmations (pointillés)

# Lignes de seuil LISA sur le panneau
ax_l.axhline(LISA_GFM_THRESHOLD / 100, color=C_SEUIL_GFM, ls="--", lw=1.2, alpha=0.85)  # Seuil GFM (normalisé)
ax_l.axhline(LISA_IFM_THRESHOLD,        color=C_SEUIL_IFM, ls="--", lw=1.2, alpha=0.85)  # Seuil IFM
ax_l.axhline(LISA_SLOPE_THRESHOLD,      color=C_SEUIL_SLP, ls="--", lw=1.2, alpha=0.85)  # Seuil pente
ax_l.axhline(1.0, color=C_OR_BRUT, ls=":", lw=1.0, alpha=0.6, label="Seuil confirmation")  # Seuil confirmations
ax_l.set_ylim(-2, 12)
ax_l.set_ylabel("Indicateurs\nLISA", color=C_TICK, fontsize=7)
ax_l.legend(loc="upper left", fontsize=6, facecolor="white", edgecolor=C_SPINE, ncol=2)

# ── Panneau fusion — configuration initiale (recréé à chaque step par cla()) ──
ax_fus.set_ylim(-0.5, 3.5)
ax_fus.set_yticks([0, 1, 1.5, 2, 3])
ax_fus.set_yticklabels(["OK", "SUSP.", "SUSP.\nSÉQ.", "PANNE\nfusion", "PANNE\nréelle"],
                        color=C_TICK, fontsize=6)
ax_fus.set_ylabel("Décision", color=C_TICK, fontsize=7)
ax_fus.set_xlabel("Heure simulée", color=C_TICK, fontsize=8)

fusion_level_history = []   # Historique du niveau numérique de fusion (0=OK, 1=SUSP, 1.5=SEQ, 2=PANNE)
real_level_history   = []   # Historique de l'état réel de la pompe (3=PANNE, 2.85=OK) pour le graphe

# ── Textes dynamiques dans le panneau CGM ──
txt_etat  = ax_cgm.text(0.99, 0.97, "", transform=ax_cgm.transAxes,
                         color=C_CGM, fontsize=13, ha="right", va="top",
                         fontfamily="monospace", fontweight="bold")  # Glycémie + état fusion

txt_info  = ax_cgm.text(0.99, 0.89, "", transform=ax_cgm.transAxes,
                         color="#444444", fontsize=8, ha="right", va="top")  # Informations détaillées

txt_cas   = ax_cgm.text(0.99, 0.82, "", transform=ax_cgm.transAxes,
                         color="#7a5c00", fontsize=8, ha="right", va="top",
                         fontfamily="monospace")  # Cas courant et écart K→L

txt_alarm = ax_cgm.text(0.5, 0.50, "", transform=ax_cgm.transAxes,
                         color=C_HAUTE, fontsize=12, ha="center", va="center",
                         fontfamily="monospace", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe5e5",
                                   edgecolor=C_HAUTE, alpha=0.0))  # Boîte d'alerte centrale (invisible par défaut)

# Titre global de la figure
fig.suptitle(
    f"Fusion Kalman + LISA v1 — Détection de panne pompe ({PATIENT_NAME})\n"
    f"LISA: score≥{LISA_SCORE_MIN} + confirmation {LISA_CONFIRMATION_STEPS} steps "
    f"({LISA_CONFIRMATION_STEPS*3}min) | Kalman: fenêtre {KALMAN_WINDOW} steps, "
    f"seuil {KALMAN_THRESHOLD}  —  Ctrl+C",
    fontsize=10, color="#111111", fontfamily="monospace"
)

failure_bands = {ax: [] for ax in AXES}   # Dictionnaire stockant les bandes colorées panne/marche par axe
fill_ref      = [None]                     # Référence à l'area fill sous la courbe CGM (liste pour mutabilité)

plt.tight_layout(rect=[0, 0, 1, 0.97])    # Ajustement automatique des marges (laisse 3% pour le suptitle)
plt.ion()                                  # Mode interactif : permet la mise à jour en temps réel du graphe
plt.show()                                 # Affiche la fenêtre graphique


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — INITIALISATION DE LA BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

s    = env.reset()         # Réinitialise l'environnement SimGlucose, retourne l'état initial
obs  = s.observation       # Observation initiale (contient obs.CGM = glycémie initiale)
done = s.done              # Indicateur de fin de simulation SimGlucose

index      = 0             # Compteur de steps
prev_state = None          # État précédent de la pompe (pour détecter les transitions)

# Instanciation des trois modules principaux
kalman = KalmanDetector(float(obs.CGM), sample_time)  # Kalman initialisé avec la glycémie de départ
lisa   = LISADetector(sample_time)                    # LISA initialisé
fusion = FusionStateMachine()                         # Machine à états fusion initialisée

# Instanciation des trackers de métriques (un par détecteur)
metrics_kalman = MetricsTracker("KALMAN", STEPS_24H)
metrics_lisa   = MetricsTracker("LISA",   STEPS_24H)
metrics_fusion = MetricsTracker("FUSION", STEPS_24H, mode="fusion")

# Enregistrement de la liste de bolus dans chaque tracker
metrics_kalman.register_bolus_steps(BOLUS_CARBS_STEPS)
metrics_lisa.register_bolus_steps(BOLUS_CARBS_STEPS)
metrics_fusion.register_bolus_steps(BOLUS_CARBS_STEPS)

# Historiques (listes) pour le graphe et le calcul final des métriques
times             = []    # Étiquettes temporelles formatées "HH:MM"
cgm_vals          = []    # Valeurs CGM à chaque step
pred_vals         = []    # Prédictions Kalman à chaque step
pump_states       = []    # État pompe (True=HS, False=OK) à chaque step
k_scores          = []    # Scores Kalman à chaque step
k_alerts_h        = []    # Alertes Kalman (bool) à chaque step
l_gfm_h           = []    # Valeurs GFM LISA à chaque step
l_ifm_h           = []    # Valeurs IFM LISA à chaque step
l_slope_h         = []    # Valeurs pente LISA à chaque step
l_consec_h        = []    # Compteurs consécutifs LISA à chaque step
l_or_brut_h       = []    # Score brut LISA (bool) à chaque step
l_alerts_h        = []    # Alertes LISA confirmées (bool) à chaque step
missed_bolus_h    = []    # Bolus manqués (bool) à chaque step
fusion_states     = []    # États fusion ("OK","SUSPICION",...) à chaque step
cas_history       = []    # Cas de fusion à chaque step
confiance_history = []    # Niveau de confiance fusion à chaque step

MAX_PTS = 80   # Nombre maximum de points affichés simultanément dans le graphe (fenêtre glissante)

# Compteurs de suivi de l'insuline
bolus_total_prevu   = 0     # Nombre total de bolus prévus
bolus_total_injecte = 0     # Nombre total de bolus effectivement injectés
bolus_total_manque  = 0     # Nombre total de bolus manqués
insuline_injectee_U = 0.0   # Insuline totale injectée en unités
insuline_manquee_U  = 0.0   # Insuline totale manquée (pompe HS) en unités
insuline_basale_U   = 0.0   # Insuline basale totale injectée en unités


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — BOUCLE PRINCIPALE DE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

STOP_REASON = "Ctrl+C"   # Raison d'arrêt par défaut (sera écrasée par "fin automatique" si applicable)

try:
    while index < MAX_SIM_STEPS:   # Boucle sur tous les steps de simulation (24h)

        # ── Mise à jour dynamique des bolus et des métriques ──
        update_bolus_steps(index, scenario, sample_time)     # Étend le planning de repas si nécessaire
        metrics_kalman.register_bolus_steps(BOLUS_CARBS_STEPS)   # Synchronise le tracker Kalman
        metrics_lisa.register_bolus_steps(BOLUS_CARBS_STEPS)     # Synchronise le tracker LISA
        metrics_fusion.register_bolus_steps(BOLUS_CARBS_STEPS)   # Synchronise le tracker Fusion

        # ── Lecture de l'état courant ──
        cgm_val    = float(obs.CGM)           # Glycémie mesurée par le CGM au step courant
        sim_min    = index * sample_time      # Temps simulé en minutes depuis le début
        time_label = (start_time + timedelta(minutes=sim_min)).strftime("%H:%M")  # Heure formatée

        carbs_for_bolus = BOLUS_CARBS_STEPS.get(index, 0)    # Glucides prévus à ce step (0 si pas de bolus)
        bolus_plan      = calculer_bolus(carbs_for_bolus, cgm_val) if carbs_for_bolus > 0 else 0.0  # Dose calculée
        failed          = pump_is_failed(index)                # True si la pompe est en panne à ce step

        # ── Détection des transitions de l'état de la pompe ──
        if prev_state is not None and failed != prev_state:    # Transition détectée
            sep = "─" * 110
            if failed:
                print(f"\n{sep}\n  [!] PANNE POMPE RÉELLE — step {index} ({time_label})\n{sep}\n")
            else:
                print(f"\n{sep}\n  [v] POMPE RÉTABLIE — step {index} ({time_label})\n{sep}\n")
        prev_state = failed   # Mémorise l'état pour le prochain step

        # ── Basal dynamique adolescent ──
        # Le débit basal est ajusté selon la glycémie pour compenser les dérives
        dynamic_basal = BASAL                            # Départ : basal de base (0.040 U/min)
        if cgm_val > 170: dynamic_basal += 0.007         # Légère hyperglycémie : +7 mU/min
        if cgm_val > 210: dynamic_basal += 0.012         # Forte hyperglycémie : +12 mU/min supplémentaires
        if cgm_val < 100: dynamic_basal -= 0.007         # Légère hypoglycémie : -7 mU/min
        if cgm_val <  85: dynamic_basal -= 0.012         # Forte hypoglycémie : -12 mU/min supplémentaires
        dynamic_basal = max(0.01, dynamic_basal)         # Minimum absolu 0.01 U/min (sécurité)

        # ── Bolus de correction si hyperglycémie sans repas prévu ──
        if bolus_plan == 0.0 and cgm_val > 200.0:
            bolus_final = min(3.0, max(0.0, (cgm_val - CGM_CIBLE) / ISF))  # Bolus correction limité à 3U
        else:
            bolus_final = bolus_plan   # Utilise le bolus repas calculé

        # ── Construction de l'action envoyée à la pompe ──
        action = (Action(basal=0.0, bolus=0.0) if failed else        # Pompe HS : aucune injection
                  Action(basal=dynamic_basal, bolus=bolus_final))     # Pompe OK : basal + bolus

        # ── Dose totale d'insuline à ce step (pour LISA) ──
        insulin_dose_step = (0.0 if failed else
                             dynamic_basal * sample_time + bolus_final)  # Basal (U/min × min = U) + bolus

        # ── Mise à jour Kalman ──
        pred, residual, k_score, k_alert = kalman.update(cgm_val)
        # Retourne : prédiction libre, résidu, score moyen, alerte confirmée

        # ── Mise à jour LISA ──
        gfm, ifm, slope, l_alert, or_brut, consec = lisa.update(cgm_val, insulin_dose_step)
        # Retourne : GFM, IFM, pente, alerte confirmée, score brut, compteur consécutifs

        # ── Mise à jour Fusion ──
        bolus_window_start = get_inter_bolus_window_start(index, BOLUS_CARBS_STEPS)  # Début de fenêtre courante
        etat_fusion, confiance, cas = fusion.update(
            k_alert, l_alert, index, cgm_val, sample_time, bolus_window_start
        )
        # Retourne : état ("OK","SUSPICION",...), confiance ("HAUTE",...), libellé du cas

        # ── Identification du détecteur responsable ──
        if k_alert and l_alert:    detecteur_txt = "Kalman + LISA"   # Les deux alertent
        elif k_alert:              detecteur_txt = "Kalman seul"      # Kalman seul
        elif l_alert:              detecteur_txt = "LISA seule"       # LISA seul
        else:                      detecteur_txt = "Aucun"            # Aucun

        # ── Envoi des notifications push ──
        if etat_fusion == "PANNE":
            notifier_panne(index)   # Notification urgente
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":
            # Calcule l'écart entre les deux détecteurs pour le message
            _ecart_seq = 0
            if (fusion._k_first_alert_step is not None and
                    fusion._l_first_alert_step is not None):
                _ecart_seq = abs(fusion._l_first_alert_step - fusion._k_first_alert_step)
            notifier_suspicion_sequentielle(index, _ecart_seq, sample_time)
        elif etat_fusion == "SUSPICION":
            notifier_suspicion(index)

        # ── Note sur la fusion passive ──
        # En v3, la fusion ne commande plus Kalman ni LISA.
        # Plus de kalman.reset(), kalman.set_freeze(), ni lisa.reset() depuis la fusion.

        # ── Mise à jour des métriques inter-bolus ──
        bolus_attendu_ce_step = index in BOLUS_CARBS_STEPS          # Bolus prévu à ce step ?
        bolus_manque_ce_step  = failed and bolus_attendu_ce_step     # Bolus prévu ET pompe HS ?
        missed_bolus_h.append(bolus_manque_ce_step)                  # Enregistre dans l'historique

        metrics_kalman.update(index, k_alert, bolus_manque_ce_step=bolus_manque_ce_step)
        metrics_lisa.update(index, l_alert, bolus_manque_ce_step=bolus_manque_ce_step)
        metrics_fusion.update_fusion(index, k_alert, l_alert, bolus_manque_ce_step)

        # ── Affichage console (tableau de suivi) ──
        conf_txt = confiance if confiance else "—"    # Affiche "—" si confiance non définie
        or_txt   = "OR" if or_brut else "  "          # Affiche "OR" si score brut LISA ≥ SCORE_MIN
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

        # ── Mise à jour des historiques ──
        times.append(time_label)              # Heure formatée
        cgm_vals.append(cgm_val)             # CGM mesuré
        pred_vals.append(pred)               # Prédiction Kalman (libre)
        pump_states.append(failed)           # État pompe
        k_scores.append(k_score)             # Score Kalman
        k_alerts_h.append(k_alert)           # Alerte Kalman
        l_gfm_h.append(gfm)                  # GFM LISA
        l_ifm_h.append(ifm)                  # IFM LISA
        l_slope_h.append(slope)              # Pente LISA
        l_consec_h.append(consec)            # Compteur consécutifs LISA
        l_or_brut_h.append(or_brut)          # Score brut LISA
        l_alerts_h.append(l_alert)           # Alerte LISA confirmée
        fusion_states.append(etat_fusion)    # État fusion
        cas_history.append(cas)              # Cas fusion
        confiance_history.append(confiance)  # Confiance fusion

        # Conversion de l'état fusion en valeur numérique pour le graphe de fusion
        fusion_level_history.append(
            2   if etat_fusion == "PANNE" else
            1.5 if etat_fusion == "SUSPICION_SEQUENTIELLE" else
            1   if etat_fusion == "SUSPICION" else 0
        )
        real_level_history.append(3 if failed else 2.85)  # Pompe HS = 3, OK = 2.85 (légèrement différent pour lisibilité)

        # ── Suivi de la comptabilité insuline ──
        if bolus_attendu_ce_step:
            bolus_total_prevu += 1                    # Incrémente le nombre de bolus prévus
            if failed:
                bolus_total_manque  += 1              # Bolus manqué (pompe HS)
                insuline_manquee_U  += bolus_plan     # Cumule l'insuline manquée
            else:
                bolus_total_injecte += 1              # Bolus injecté normalement
                insuline_injectee_U += bolus_final    # Cumule l'insuline injectée
        elif not failed and bolus_final > 0:
            insuline_injectee_U += bolus_final        # Bolus de correction sans repas prévu
        if not failed:
            insuline_basale_U += dynamic_basal * sample_time  # Cumule l'insuline basale

        # ─────────────────────────────────────────────────────────────────────
        # SOUS-SECTION : Mise à jour du graphe (fenêtre glissante MAX_PTS)
        # ─────────────────────────────────────────────────────────────────────

        # Extraction des MAX_PTS derniers points pour l'affichage
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

        x      = list(range(len(disp_times)))           # Index de l'axe X (0 à MAX_PTS-1)
        offset = max(0, len(times) - MAX_PTS)           # Décalage pour retrouver les steps réels

        # ── Mise à jour des courbes du graphe ──
        line_cgm.set_data(x, disp_cgm)     # Courbe CGM
        line_pred.set_data(x, disp_pred)   # Courbe prédiction Kalman
        line_kscore.set_data(x, disp_kscore)  # Score Kalman
        line_gfm.set_data(x, [v / 100.0 for v in disp_gfm])   # GFM /100 pour la mise à l'échelle
        line_ifm.set_data(x, disp_ifm)       # IFM
        line_slope.set_data(x, disp_slope)   # Pente
        line_consec.set_data(x, [v / max(1, LISA_CONFIRMATION_STEPS) for v in disp_consec])  # Normalisé [0,1]

        # ── Area fill sous la courbe CGM ──
        if fill_ref[0] is not None:
            fill_ref[0].remove()            # Supprime l'ancien fill avant d'en créer un nouveau
        fill_ref[0] = ax_cgm.fill_between(x, disp_cgm, alpha=0.08, color=C_CGM, zorder=2)  # Nouveau fill

        # ── Panneau fusion : barres colorées par état ──
        ax_fus.cla()                    # Efface tout le panneau fusion pour le redessiner
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

        # Dessin des barres colorées (une par step) selon l'état de fusion
        for i_f, (fv, rv, fc) in enumerate(zip(disp_fus, disp_real, disp_conf)):
            if fv == 2:     color_f = C_HAUTE     # Rouge : PANNE confirmée
            elif fv == 1.5: color_f = C_SEQ       # Ambre : SUSPICION SÉQUENTIELLE
            elif fv == 1:   color_f = "#e6a817"   # Jaune : SUSPICION simple
            else:           color_f = "#c8e6c9"   # Vert pâle : OK
            ax_fus.bar(i_f, fv + 0.5, bottom=-0.5, width=1.0,
                       color=color_f, alpha=0.80, align="edge")   # Barre de fusion
            ax_fus.plot(i_f + 0.5, rv, marker="_", ms=10,
                        color=C_BOLUS_FAIL, lw=2, alpha=0.9)      # Marqueur "_" = état pompe réel

        # ── Synchronisation des limites X sur tous les axes ──
        for ax in AXES:
            ax.set_xlim(-1, len(disp_times))

        # ── Étiquettes temporelles de l'axe X (toutes les 6 steps) ──
        ticks = list(range(0, len(disp_times), 6))
        ax_fus.set_xticks(ticks)
        ax_fus.set_xticklabels(
            [disp_times[i] for i in ticks if i < len(disp_times)],
            fontsize=7, color="#64748b"
        )
        # Cache les étiquettes sur les autres axes (ils partagent le même X via sharex)
        for ax in [ax_cgm, ax_k, ax_l]:
            ax.set_xticks(ticks)
            ax.set_xticklabels([], visible=False)

        # ── Bandes de fond panne/marche ──
        # Supprime les anciennes bandes
        for ax in AXES:
            for band in failure_bands[ax]:
                try: band.remove()
                except Exception: pass
            failure_bands[ax].clear()

        # Recrée les bandes par segments continus (regroupement des steps de même état)
        i_seg = 0
        while i_seg < len(disp_states):
            seg_state = disp_states[i_seg]; j_seg = i_seg
            while j_seg < len(disp_states) and disp_states[j_seg] == seg_state:
                j_seg += 1             # Trouve la fin du segment
            c_band = C_BAND_FAIL if seg_state else C_BAND_OK    # Couleur selon l'état
            a_band = 0.45 if seg_state else 0.35                 # Opacité plus forte pour les pannes
            for ax in AXES:
                b = ax.axvspan(i_seg - 0.5, j_seg - 0.5, alpha=a_band, color=c_band, zorder=1)
                failure_bands[ax].append(b)    # Mémorise pour pouvoir supprimer au prochain step
            i_seg = j_seg

        # ── Marqueurs de repas ──
        mx, my = [], []
        for r in scenario.repas:
            r_step = r.get("step")
            if r_step is None: continue
            vi = r_step - offset                          # Index dans la fenêtre d'affichage
            if 0 <= vi < len(disp_times):
                mx.append(vi); my.append(disp_cgm[vi] + 18)  # Positionne le marqueur 18 mg/dL au-dessus du CGM
        sc_meals.set_offsets(np.c_[mx, my] if mx else np.empty((0, 2)))

        # ── Marqueurs de bolus (OK en bleu, manqués en rouge) ──
        ok_x, ok_y, fx, fy = [], [], [], []
        for step_b in BOLUS_STEPS:
            vi = step_b - offset
            if 0 <= vi < len(disp_times):
                if pump_is_failed(step_b): fx.append(vi); fy.append(disp_cgm[vi] - 18)   # Bolus manqué
                else:                      ok_x.append(vi); ok_y.append(disp_cgm[vi] - 18)  # Bolus OK
        sc_bolus_ok.set_offsets(np.c_[ok_x, ok_y] if ok_x else np.empty((0, 2)))
        sc_bolus_fail.set_offsets(np.c_[fx, fy] if fx else np.empty((0, 2)))

        # ── Marqueurs d'alertes ──
        ph_x, ph_y, su_x, su_y, sq_x, sq_y = [], [], [], [], [], []
        for i_a, (fs, fc) in enumerate(zip(disp_fusion_states, disp_conf)):
            if fs == "PANNE":
                ph_x.append(i_a); ph_y.append(disp_cgm[i_a] + 30)    # Losange rouge PANNE
            elif fs == "SUSPICION_SEQUENTIELLE":
                sq_x.append(i_a); sq_y.append(disp_cgm[i_a] + 25)    # Carré ambre SUSP.SEQ.
            elif fs == "SUSPICION":
                su_x.append(i_a); su_y.append(disp_cgm[i_a] + 20)    # Cercle vert SUSPICION
        sc_panne_haute.set_offsets(np.c_[ph_x, ph_y] if ph_x else np.empty((0, 2)))
        sc_susp_seq.set_offsets(np.c_[sq_x, sq_y] if sq_x else np.empty((0, 2)))
        sc_suspicion.set_offsets(np.c_[su_x, su_y] if su_x else np.empty((0, 2)))

        # ── Mise à jour des textes dynamiques ──
        # Couleur du texte d'état selon la gravité
        if etat_fusion == "PANNE":                     col_etat = C_HAUTE
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":  col_etat = C_SEQ
        elif etat_fusion == "SUSPICION":               col_etat = "#e6a817"
        else:                                          col_etat = C_CGM
        txt_etat.set_text(f"{cgm_val:.0f} mg/dL  [{etat_fusion}]")   # "120 mg/dL  [OK]"
        txt_etat.set_color(col_etat)

        # Texte d'information détaillée
        txt_info.set_text(
            f"{time_label} | step {index} | "
            f"Kalman {'✓' if k_alert else '·'}  LISA {'✓' if l_alert else '·'} "
            f"(Score≥{LISA_SCORE_MIN}: {'✓' if or_brut else '·'} ×{consec}/{LISA_CONFIRMATION_STEPS})  |  "
            f"GFM={gfm:.1f}  IFM={ifm:.3f}  Pente={slope:.2f}  |  "
            f"Réel : {'PANNE' if failed else 'OK'}"
        )

        # Texte du cas courant avec écart K→L
        ecart_txt = ""
        if fusion.ecarts:
            last_step, last_ecart, last_conf = fusion.ecarts[-1]   # Dernier écart enregistré
            if last_ecart is not None:
                signe     = "+" if last_ecart > 0 else ""
                ecart_txt = (f"Écart K→L : {signe}{last_ecart} steps "
                             f"({signe}{last_ecart*sample_time:.0f} min)")
            else:
                ecart_txt = f"Un seul détecteur ({last_conf})"
        txt_cas.set_text(f"Cas : {cas}\nConfiance : {confiance if confiance else '—'}\n{ecart_txt}")

        # ── Boîte d'alerte centrale ──
        if etat_fusion == "PANNE":
            txt_alarm.set_text("PANNE CONFIRMÉE — Kalman + LISA")   # Texte rouge
            txt_alarm.set_color(C_HAUTE)
            txt_alarm.get_bbox_patch().set_edgecolor(C_HAUTE)
            txt_alarm.get_bbox_patch().set_facecolor("#ffe5e5")    # Fond rosé
            txt_alarm.get_bbox_patch().set_alpha(0.90)
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":
            if (fusion._k_first_alert_step is not None and
                    fusion._l_first_alert_step is not None):
                _ordre = "K → L" if fusion._k_first_alert_step <= fusion._l_first_alert_step else "L → K"
            else:
                _ordre = "K ↔ L"
            txt_alarm.set_text(f"SUSPICION SÉQUENTIELLE — {_ordre}")   # Texte ambre avec ordre
            txt_alarm.set_color(C_SEQ)
            txt_alarm.get_bbox_patch().set_edgecolor(C_SEQ)
            txt_alarm.get_bbox_patch().set_facecolor("#fff3cd")    # Fond jaune pâle
            txt_alarm.get_bbox_patch().set_alpha(0.90)
        elif etat_fusion == "SUSPICION":
            txt_alarm.set_text(f"SUSPICION — {detecteur_txt}")   # Texte jaune avec détecteur actif
            txt_alarm.set_color("#7a5c00")
            txt_alarm.get_bbox_patch().set_edgecolor("#e6a817")
            txt_alarm.get_bbox_patch().set_facecolor("#fff8e1")    # Fond crème
            txt_alarm.get_bbox_patch().set_alpha(0.85)
        else:
            txt_alarm.set_text("")                              # Cache la boîte d'alerte
            txt_alarm.get_bbox_patch().set_alpha(0.0)

        # ── Rendu du graphe ──
        try:
            fig.canvas.draw_idle()    # Demande un redessinage asynchrone (non bloquant)
            fig.canvas.flush_events() # Force le traitement des événements graphiques
        except Exception:
            pass   # Ignore les erreurs de rendu (ex : fenêtre fermée)

        time.sleep(STEP_DELAY)    # Pause optionnelle entre les steps (0.0 = pas de pause)

        # ── Passage au step suivant ──
        if done:
            # SimGlucose a signalé la fin de simulation : réinitialise l'environnement
            print(f"\n[INFO] Reset environnement (step {index})\n")
            p2  = T1DPatient.withName(PATIENT_NAME)
            s2  = CGMSensor.withName("Dexcom", seed=index)
            pu2 = InsulinPump.withName("Insulet")
            sc2 = scenario
            sc2.start_time = start_time + timedelta(minutes=index * sample_time)  # Met à jour l'heure de départ
            env2 = T1DSimEnv(patient=p2, sensor=s2, pump=pu2, scenario=sc2)
            s    = env2.reset()                    # Réinitialise le nouvel environnement
            env  = env2                            # Remplace l'environnement courant
            kalman = KalmanDetector(float(s.observation.CGM), sample_time)  # Réinit Kalman
            lisa   = LISADetector(sample_time)                               # Réinit LISA
        else:
            s = env.step(action)    # Exécute le step dans SimGlucose avec l'action (basal + bolus)

        obs  = s.observation    # Récupère la nouvelle observation (CGM, etc.)
        done = s.done           # Vérifie si SimGlucose a terminé
        index += 1              # Incrémente le compteur de steps

    # ── Fin automatique de la simulation (MAX_SIM_STEPS atteint) ──
    STOP_REASON = f"fin automatique {SIM_DURATION_HOURS} h"
    raise KeyboardInterrupt   # Déclenche le même chemin de fin que Ctrl+C

except KeyboardInterrupt:
    # ── Affichage du résumé final des métriques à l'arrêt ──
    afficher_resume_metriques_24h(
        BOLUS_CARBS_STEPS,
        missed_bolus_h,
        k_alerts_h,
        l_alerts_h,
        index,
    )

plt.ioff()   # Désactive le mode interactif matplotlib (termine l'affichage en temps réel)