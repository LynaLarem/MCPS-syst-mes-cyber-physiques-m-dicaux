# ============================================================
# fusionadulte_commente.py
# Code original : fusionadulte.py
# Objectif : Simulation continue de détection de panne de pompe à insuline
# chez un patient adulte diabétique de type 1 (DT1).
# Deux algorithmes de détection sont fusionnés :
#   - Kalman-CUSUM : filtre de Kalman + test statistique CUSUM
#   - LISA         : algorithme à base de fenêtres glissantes (GFM, IFM, Pente)
# Une machine à états (FusionStateMachine) combine les deux décisions.
# Des notifications sont envoyées via ntfy.sh.
# ============================================================

# ── Imports standard Python ──────────────────────────────────────────────────

import time          # Pour les pauses (time.sleep) entre chaque step de simulation
import threading     # Pour envoyer les notifications ntfy en parallèle (non-bloquant)
import urllib.request  # Pour faire des requêtes HTTP vers ntfy.sh (sans dépendances tierces)
import numpy as np   # Calculs vectoriels/matriciels (Kalman, moyennes, etc.)

# Configuration du backend graphique AVANT d'importer pyplot :
# "TkAgg" utilise Tkinter pour ouvrir une fenêtre graphique interactive.
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt              # Tracé du graphique en temps réel
import matplotlib.gridspec as gridspec       # Pour diviser la figure en sous-panneaux de tailles différentes
from matplotlib.lines import Line2D          # Pour créer des éléments de légende personnalisés (lignes)
from matplotlib.patches import Patch         # Pour créer des éléments de légende personnalisés (patches/zones)
from datetime import datetime, timedelta     # Pour gérer les horodatages simulés (heure de début + écoulé)
from collections import namedtuple           # Pour créer le type Action (basal, bolus) de manière lisible

# ── Imports de simglucose ────────────────────────────────────────────────────
# simglucose est un simulateur open-source de patient DT1 (Diabète de Type 1)

from simglucose.simulation.env import T1DSimEnv           # Environnement de simulation principal
from simglucose.simulation.scenario import Scenario        # Classe de base pour les scénarios de repas
from simglucose.simulation.scenario import Action as MealAction  # Action repas (grammes de glucides)
from simglucose.patient.t1dpatient import T1DPatient       # Modèle physio du patient DT1
from simglucose.sensor.cgm import CGMSensor                # Capteur de glycémie en continu (CGM)
from simglucose.actuator.pump import InsulinPump           # Pompe à insuline


# ============================================================
# SECTION 1 — CONFIGURATION DU PATIENT ADULTE
# ============================================================

# Nom du patient dans la base simglucose (profil physiologique adulte prédéfini)
PATIENT_NAME = "adult#001"

# Planning de repas sur une "journée simulée" (cycle de 600 minutes = 10 h)
# Chaque dictionnaire contient :
#   - "min_in_day" : minute dans la journée où le repas survient
#   - "carbs"      : quantité de glucides en grammes
#   - "label"      : étiquette affichée sur le graphique
REPAS_JOURNEE = [
    {"min_in_day":  60, "carbs": 50, "label": "Petit-dej\n50g"},   # Repas 1 à 60 min
    {"min_in_day": 180, "carbs": 70, "label": "Dejeuner\n70g"},    # Repas 2 à 180 min
    {"min_in_day": 300, "carbs": 25, "label": "Gouter\n25g"},      # Repas 3 à 300 min
    {"min_in_day": 420, "carbs": 60, "label": "Diner\n60g"},       # Repas 4 à 420 min
    {"min_in_day": 540, "carbs": 20, "label": "Collation\n20g"},   # Repas 5 à 540 min
]

# Paramètres du calcul de bolus insuline :
ICR       = 12.0   # Insulin-to-Carb Ratio : 1 unité d'insuline couvre 12 g de glucides
ISF       = 50.0   # Insulin Sensitivity Factor : 1 unité abaisse la glycémie de 50 mg/dL
CGM_CIBLE = 120.0  # Glycémie cible (mg/dL) pour le bolus de correction
BOLUS_MIN = 0.3    # Dose minimale de bolus autorisée (unités)
BOLUS_MAX = 15.0   # Dose maximale de bolus autorisée (unités) — sécurité anti-surdosage


def calculer_bolus(carbs, cgm_val):
    """
    Calcule la dose de bolus pré-repas selon la règle standard :
      bolus = bolus_repas + bolus_correction
    avec :
      bolus_repas      = carbs / ICR
      bolus_correction = (CGM - CGM_CIBLE) / ISF (si CGM >= 80)

    Sécurités :
      - Si CGM < 70 mg/dL (hypoglycémie) → pas de bolus (retour 0.0)
      - Le résultat est clippé entre BOLUS_MIN et BOLUS_MAX
    """
    if cgm_val < 70.0:
        return 0.0  # Hypoglycémie détectée : on n'injecte pas d'insuline
    bolus_repas      = carbs / ICR  # Part repas : proportionnelle aux glucides
    # Part correction : non nulle seulement si CGM assez élevé
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0
    # Clipping entre min et max, arrondi à 2 décimales
    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)


# Durée d'un cycle journalier simulé (600 min = 10 h avec 5 repas)
MINUTES_PAR_JOUR = 600
REPAS            = []  # Liste globale des repas, remplie dynamiquement


def build_repas_for_horizon(sample_time, horizon_steps=10000):
    """
    Génère la liste complète des repas pour un horizon de simulation donné.
    Le planning se répète par cycle de MINUTES_PAR_JOUR minutes.

    Paramètres :
        sample_time    : intervalle entre deux mesures CGM (en minutes)
        horizon_steps  : nombre de steps à couvrir à l'avance

    Retourne une liste de dictionnaires repas avec clés :
        "min"   → minute absolue dans la simulation
        "carbs" → glucides (g)
        "bolus" → sera rempli plus tard (None pour l'instant)
        "label" → étiquette graphique
    """
    repas     = []
    total_min = horizon_steps * sample_time   # Durée totale en minutes
    nb_jours  = int(total_min / MINUTES_PAR_JOUR) + 2  # Nombre de cycles + marge
    for jour in range(nb_jours):
        base_min = jour * MINUTES_PAR_JOUR    # Offset de début de ce cycle
        for r in REPAS_JOURNEE:
            repas.append({
                "min"   : base_min + r["min_in_day"],  # Minute absolue du repas
                "carbs" : r["carbs"],
                "bolus" : None,                        # Sera calculé plus tard
                "label" : r["label"],
            })
    return repas


# ============================================================
# SECTION 2 — PARAMÈTRES DE SIMULATION DE PANNE
# ============================================================

PANNE_DUREE      = 30    # Nombre de steps pendant lesquels la pompe est en panne
MARCHE_DUREE     = 25    # Nombre de steps pendant lesquels la pompe fonctionne
PANNE_START_STEP = 10    # Step à partir duquel les cycles panne/marche commencent
BASAL            = 0.050 # Débit basal nominal (unités/min) quand la pompe fonctionne
PREBOLUS_STEPS   = 8     # Nombre de steps d'anticipation pour le pré-bolus (avant le repas)
STEP_DELAY       = 1.5   # Délai en secondes entre deux steps (contrôle la vitesse de simulation)

# None = simulation infinie (arrêt uniquement avec Ctrl+C)
SIM_DURATION_HOURS = None

# Définition d'un type nommé pour les actions de la pompe (plus lisible qu'un tuple simple)
Action = namedtuple("Action", ["basal", "bolus"])


# ============================================================
# SECTION 3 — PARAMÈTRES DE L'ALGORITHME LISA (adulte)
# ============================================================
# LISA = algorithme de détection basé sur des fenêtres glissantes
# Il surveille la montée de glycémie et la baisse d'insuline active.

LISA_LW                 = 15       # Taille de la longue fenêtre (Long Window) en steps
LISA_SW                 = 8        # Taille de la courte fenêtre (Short Window) en steps
LISA_GFM_THRESHOLD      = 755.739902  # Seuil de l'indicateur GFM (Glucose Flux Metric)
LISA_IFM_THRESHOLD      = 0.027636   # Seuil de l'indicateur IFM (Insulin Flux Metric)
LISA_SLOPE_THRESHOLD    = 2.128712   # Seuil de la pente de glycémie (mg/dL/min)
LISA_SCORE_MIN          = 3          # Score minimum pour déclencher une alerte brute
                                     # (GFM vaut 2 pts, IFM et Pente valent 1 pt chacun)
LISA_PIE_ALPHA          = 0.85       # Facteur de lissage exponentiel pour l'insuline active (PIE)
LISA_CONFIRMATION_STEPS = 3          # Nombre de steps consécutifs à score≥min pour confirmer l'alerte


# ============================================================
# SECTION 4 — PARAMÈTRES DU DÉTECTEUR KALMAN-CUSUM (adulte)
# ============================================================
# Le filtre de Kalman prédit la glycémie.
# Le CUSUM accumule les résidus positifs (CGM > prédiction) pour détecter une dérive.

CUSUM_RES_WINDOW     = 3         # Fenêtre de lissage des résidus avant le CUSUM (en steps)
CUSUM_K              = 97.396446 # Seuil de référence du CUSUM (valeur calibrée)
CUSUM_H              = 389.585782 # Seuil d'alarme du CUSUM (valeur calibrée)
KALMAN_CONFIRM_STEPS = 4         # Nombre de steps consécutifs au-dessus de CUSUM_H pour confirmer

# Matrice de covariance du bruit de processus Q (incertitude sur l'évolution du modèle)
# État = [glycémie, vitesse de variation]
Q_PROFILE = np.array([[2.5,  0.0],   # Variance sur la glycémie
                       [0.0,  0.25]]) # Variance sur la vitesse

# Matrice de covariance du bruit de mesure R (bruit du capteur CGM)
R_PROFILE = np.array([[25.0]])        # Variance de mesure du CGM (en mg²/dL²)


# ============================================================
# SECTION 5 — PARAMÈTRE DE LA MACHINE À ÉTATS FUSION
# ============================================================

# Fenêtre temporelle (en steps) pour détecter des alertes "séquentielles" inter-bolus
# ~120 min avec sample_time=3 min
SEQ_WINDOW_STEPS = 40


# ============================================================
# SECTION 6 — NOTIFICATIONS NTFY.SH
# ============================================================
# ntfy.sh est un service de notifications push gratuit.
# Trois canaux sont utilisés selon la gravité de l'alerte.

NTFY_SUSPICION_URL     = "https://ntfy.sh/pompe_suspicion_detecteur"
NTFY_SUSPICION_SEQ_URL = "https://ntfy.sh/pompe_suspicion_sequentielles"
NTFY_PANNE_URL         = "https://ntfy.sh/pompe_panne_confirme"

# Anti-spam : délai minimum (en steps) entre deux notifications du même type
NTFY_COOLDOWN_STEPS = 10

# Mémorisation du dernier step où chaque type de notification a été envoyé
# Initialisés à -999 pour que la première notification soit toujours envoyée
_ntfy_last_suspicion     = -999
_ntfy_last_suspicion_seq = -999
_ntfy_last_panne         = -999


def _send_ntfy(url, title, message, priority="default", tags="rotating_light,warning"):
    """
    Envoie une notification HTTP POST vers ntfy.sh dans un thread séparé.
    Le thread est "daemon=False" pour garantir que la notification est envoyée
    même si le programme principal se termine.

    Paramètres :
        url      : URL du canal ntfy.sh
        title    : Titre de la notification (header "Title")
        message  : Corps du message (UTF-8)
        priority : Priorité ntfy (default / high / urgent)
        tags     : Icônes affichées dans la notification (séparées par des virgules)
    """
    def _do_send():
        try:
            data = message.encode("utf-8")                    # Encodage du message en bytes
            req  = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Title",        title)             # Titre de la notification
            req.add_header("Priority",     priority)          # Niveau d'urgence
            req.add_header("Tags",         tags)              # Icônes emoji
            req.add_header("Content-Type", "text/plain; charset=utf-8")
            with urllib.request.urlopen(req, timeout=8) as resp:
                print(f"[NTFY] HTTP {resp.status} -> {url.split('/')[-1]}")
        except Exception as e:
            print(f"[NTFY] Erreur envoi notification : {e}")
    # Lancement dans un thread séparé pour ne pas bloquer la simulation
    threading.Thread(target=_do_send, daemon=False).start()


def notifier_suspicion(step):
    """
    Envoie une notification de SUSPICION (un seul détecteur actif).
    Le cooldown (NTFY_COOLDOWN_STEPS) évite le spam si la suspicion persiste.
    """
    global _ntfy_last_suspicion
    if step - _ntfy_last_suspicion >= NTFY_COOLDOWN_STEPS:
        _ntfy_last_suspicion = step  # Mise à jour du dernier envoi
        _send_ntfy(
            NTFY_SUSPICION_URL,
            "Suspicion pompe [ADULTE]",
            "Anomalie detectee par un seul detecteur - surveillance renforcee.",
            priority="high",
            tags="warning,mag"
        )
        print(f"[NTFY] Notification SUSPICION envoyee (step {step})")


def notifier_suspicion_sequentielle(step, ecart_steps, sample_time):
    """
    Envoie une notification de SUSPICION SÉQUENTIELLE (les deux détecteurs ont
    alerté successivement dans la même fenêtre inter-bolus).
    Inclut l'écart en minutes entre les deux alertes.
    """
    global _ntfy_last_suspicion_seq
    if step - _ntfy_last_suspicion_seq >= NTFY_COOLDOWN_STEPS:
        _ntfy_last_suspicion_seq = step
        ecart_min = abs(ecart_steps) * sample_time  # Conversion steps → minutes
        _send_ntfy(
            NTFY_SUSPICION_SEQ_URL,
            "Suspicion elevee - alertes successives [ADULTE]",
            f"Les deux detecteurs ont alerte successivement (ecart : {ecart_min:.0f} min).",
            priority="high",
            tags="warning,rotating_light"
        )
        print(f"[NTFY] Notification SUSPICION SEQUENTIELLE envoyee (step {step})")


def notifier_panne(step):
    """
    Envoie une notification de PANNE CONFIRMÉE (Kalman ET LISA alertent simultanément).
    Priorité "urgent" → notification sonore sur mobile.
    """
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


# ============================================================
# SECTION 7 — UTILITAIRES DE SIMULATION
# ============================================================

def pump_is_failed(step):
    """
    Détermine si la pompe est en panne à un step donné.
    La panne suit un cycle périodique : PANNE_DUREE steps HS, MARCHE_DUREE steps OK.
    Le cycle démarre à PANNE_START_STEP.

    Exemple avec PANNE_START_STEP=10, PANNE_DUREE=30, MARCHE_DUREE=25 :
        steps  0-9  → OK (avant le début des pannes)
        steps 10-39 → PANNE (30 steps)
        steps 40-64 → OK (25 steps)
        steps 65-94 → PANNE (30 steps)
        etc.
    """
    if step < PANNE_START_STEP:
        return False  # Période de démarrage : pompe toujours OK
    # Position dans le cycle courant
    return (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE) < PANNE_DUREE


class AdultScenario(Scenario):
    """
    Scénario de simulation pour un patient adulte.
    Hérite de Scenario (simglucose) et fournit les apports de glucides
    à chaque step de la simulation.
    """

    def __init__(self, start_time, repas, sample_time):
        self.start_time  = start_time   # Datetime de début de la simulation
        self.repas       = repas        # Liste des repas (dictionnaires)
        self.sample_time = sample_time  # Durée d'un step en minutes
        # Dictionnaire {step → carbs} pour accès rapide (O(1))
        self.meal_steps  = {int(r["min"] / sample_time): r["carbs"] for r in repas}

    def get_action(self, t):
        """
        Appelé par simglucose à chaque step.
        Retourne un objet MealAction avec la quantité de glucides à ce step (0 si pas de repas).

        Paramètre :
            t : datetime courant de la simulation
        """
        # Calcul du nombre de minutes écoulées depuis le début
        elapsed = (t - self.start_time).total_seconds() / 60
        # Conversion en numéro de step
        step    = int(round(elapsed / self.sample_time))
        # Retourne les glucides de ce step (0 si aucun repas prévu)
        return MealAction(meal=self.meal_steps.get(step, 0))

    def reset(self):
        """Méthode requise par l'interface Scenario (ici sans action)."""
        pass

    def extend_if_needed(self, current_step, sample_time, horizon=500):
        """
        Étend dynamiquement le planning de repas si l'horizon courant est atteint.
        Cela permet une simulation infinie sans planifier tous les repas à l'avance.

        Paramètres :
            current_step : step actuel de la simulation
            sample_time  : durée d'un step (minutes)
            horizon      : nombre de steps à anticiper
        """
        # Step maximal actuellement planifié
        max_step = max(self.meal_steps.keys()) if self.meal_steps else 0
        if current_step + horizon > max_step:
            last_min      = max_step * sample_time
            nb_jours_supp = int((current_step + horizon) * sample_time / MINUTES_PAR_JOUR) + 2
            for jour in range(nb_jours_supp):
                base_min = last_min + (jour + 1) * MINUTES_PAR_JOUR
                for r in REPAS_JOURNEE:
                    new_min  = base_min + r["min_in_day"]
                    new_step = int(new_min / sample_time)
                    if new_step not in self.meal_steps:    # Évite les doublons
                        self.meal_steps[new_step] = r["carbs"]
                        self.repas.append({
                            "min"  : new_min,
                            "carbs": r["carbs"],
                            "bolus": None,
                            "label": r["label"],
                            "step" : new_step,
                        })


# ============================================================
# SECTION 8 — DÉTECTEUR LISA (adulte)
# ============================================================
# LISA surveille l'évolution de la glycémie et de l'insuline active
# via trois indicateurs calculés sur des fenêtres glissantes.

class LISADetector:
    """
    Détecteur LISA (Long/Short window Insulin and glucose Analysis).

    Indicateurs calculés à chaque step :
        GFM  (Glucose Flux Metric)  : mesure la hausse cumulée de la glycémie
        IFM  (Insulin Flux Metric)  : mesure la chute de l'insuline active (PIE)
        Pente                       : vitesse de montée de la glycémie (court terme)

    Score = 2×(GFM>seuil) + 1×(IFM>seuil) + 1×(Pente>seuil)
    Alerte brute si score ≥ LISA_SCORE_MIN
    Alerte confirmée si score_brut ≥ min pendant LISA_CONFIRMATION_STEPS steps consécutifs
    """

    def __init__(self, sample_time):
        self.sample_time     = sample_time  # Durée d'un step (minutes)
        self.cgm_history     = []           # Historique des mesures CGM
        self.pie_history     = []           # Historique de l'insuline active lissée (PIE)
        self._pie_current    = 0.0          # Valeur courante de la PIE (Passive Insulin Estimation)
        self._gfm_current    = 0.0          # Valeur courante du GFM cumulatif
        self._sw_prev        = None         # Moyenne SW du step précédent (pour calcul de pente)
        self._or_consecutive = 0            # Compteur de steps consécutifs avec score brut ≥ min
        self._confirmed      = False        # État de confirmation de l'alerte LISA

    def update(self, cgm_val, insulin_dose):
        """
        Met à jour le détecteur LISA avec une nouvelle mesure CGM et la dose d'insuline.

        Retourne un tuple (gfm, ifm, slope, confirmed, score_signal, or_consecutive) :
            gfm           : valeur actuelle du GFM
            ifm           : valeur actuelle de l'IFM
            slope         : pente de la glycémie (court terme)
            confirmed     : True si l'alerte est confirmée (N steps consécutifs)
            score_signal  : True si le score brut ≥ LISA_SCORE_MIN
            or_consecutive: compteur de steps consécutifs à score élevé
        """
        dt = self.sample_time

        # ── Mise à jour de la PIE (lissage exponentiel de l'insuline active) ──
        # PIE_t = α × PIE_{t-1} + dose_t
        # α < 1 assure la décroissance naturelle de l'insuline active
        self._pie_current = LISA_PIE_ALPHA * self._pie_current + insulin_dose
        self.pie_history.append(self._pie_current)
        self.cgm_history.append(cgm_val)

        # Pas assez de données : fenêtre longue non encore remplie
        if len(self.cgm_history) < LISA_LW + 1:
            self._or_consecutive = 0
            self._confirmed      = False
            return 0.0, 0.0, 0.0, False, False, 0

        # ── Calcul des moyennes sur les deux fenêtres ─────────────────────────
        cgm_arr    = np.array(self.cgm_history[-LISA_LW:])  # Derniers LISA_LW points
        pie_arr    = np.array(self.pie_history[-LISA_LW:])
        glucose_lw = float(np.mean(cgm_arr))                  # Moyenne glycémie fenêtre longue
        glucose_sw = float(np.mean(self.cgm_history[-LISA_SW:]))  # Moyenne glycémie fenêtre courte
        pie_lw     = float(np.mean(pie_arr))                   # Moyenne PIE fenêtre longue
        pie_sw     = float(np.mean(self.pie_history[-LISA_SW:]))  # Moyenne PIE fenêtre courte

        # ── GFM : intégrale cumulative de la hausse de glycémie ───────────────
        # diff = glucose_sw - glucose_lw > 0 signifie que la glycémie monte récemment
        diff = glucose_sw - glucose_lw
        if diff > 0:
            # Accumulation du flux positif (montée de glycémie)
            self._gfm_current = self._gfm_current + diff * dt
        else:
            # Réinitialisation si la glycémie ne monte plus (ou redescend)
            self._gfm_current = 0.0
        gfm = self._gfm_current

        # ── IFM : chute relative de l'insuline active ─────────────────────────
        # IFM > 0 si l'insuline active diminue (fenêtre longue > fenêtre courte)
        # Normalisé par pie_lw pour être sans unité
        ifm = max(0.0, (pie_lw - pie_sw) / pie_lw) if pie_lw > 1e-6 else 0.0

        # ── Pente de la glycémie (court terme) ────────────────────────────────
        # Différence entre la moyenne SW actuelle et la précédente, normalisée par dt
        slope         = (glucose_sw - self._sw_prev) / dt if self._sw_prev is not None else 0.0
        self._sw_prev = glucose_sw  # Mémorisation pour le prochain step

        # ── Score LISA pondéré ────────────────────────────────────────────────
        # GFM contribue 2 points (indicateur principal), IFM et Pente 1 point chacun
        lisa_score = (2 * int(gfm   > LISA_GFM_THRESHOLD) +
                      1 * int(ifm   > LISA_IFM_THRESHOLD) +
                      1 * int(slope > LISA_SLOPE_THRESHOLD))

        score_signal = lisa_score >= LISA_SCORE_MIN  # True si le seuil minimal est atteint

        # ── Filtre de confirmation (N steps consécutifs) ──────────────────────
        if score_signal:
            self._or_consecutive += 1     # Incrémentation du compteur
        else:
            self._or_consecutive = 0      # Remise à zéro si le score redescend
            self._confirmed      = False  # L'alerte n'est plus confirmée

        # Confirmation définitive après LISA_CONFIRMATION_STEPS steps consécutifs
        if self._or_consecutive >= LISA_CONFIRMATION_STEPS:
            self._confirmed = True

        return gfm, ifm, slope, self._confirmed, score_signal, self._or_consecutive

    def reset(self):
        """Réinitialise l'état interne du détecteur LISA (sans effacer l'historique)."""
        self._gfm_current    = 0.0
        self._or_consecutive = 0
        self._confirmed      = False


# ============================================================
# SECTION 9 — DÉTECTEUR KALMAN-CUSUM (adulte)
# ============================================================

class KalmanDetector:
    """
    Détecteur basé sur un filtre de Kalman + test CUSUM positif.

    Principe :
    1. Le filtre de Kalman prédit la glycémie au step suivant.
    2. Si la mesure CGM dépasse la prédiction (résidu positif), le filtre est "gelé"
       (il n'est plus mis à jour avec la mesure réelle, pour ne pas masquer la dérive).
    3. Le CUSUM accumule les résidus positifs (lissés sur CUSUM_RES_WINDOW steps).
    4. Si le score CUSUM dépasse CUSUM_H pendant KALMAN_CONFIRM_STEPS steps → alerte confirmée.
    5. Dégel automatique si le CGM redescend pendant N_DEGEL steps consécutifs.
    6. Fusion passive : le filtre ne reçoit aucune commande externe (gel/reset).

    État du filtre de Kalman :
        x = [glycémie_estimée, vitesse_de_variation] (vecteur 2D)
        P = matrice de covariance de l'état (incertitude)
    """

    N_DEGEL = 3              # Nombre de steps descendants pour déclencher le dégel
    SLOPE_LOOKBACK_STEPS = 3  # Fenêtre de calcul de la pente CGM (pour le dégel)

    def __init__(self, first_cgm, sample_time):
        """
        Initialise le filtre de Kalman avec la première mesure CGM.

        Paramètres :
            first_cgm   : première valeur CGM (mg/dL)
            sample_time : durée d'un step (minutes)
        """
        self.dt = sample_time                        # Pas de temps

        # État initial : glycémie = first_cgm, vitesse = 0
        self.x  = np.array([[float(first_cgm)], [0.0]])

        # Matrice de covariance initiale (incertitude assez large au démarrage)
        self.P  = np.eye(2) * 10.0

        # Matrice de transition : x_{t+1} = F * x_t (modèle cinématique : position + vitesse)
        # glycémie_{t+1} = glycémie_t + vitesse_t * dt
        # vitesse_{t+1}  = vitesse_t (constante entre deux steps)
        self.F  = np.array([[1, self.dt], [0, 1]])

        # Matrice d'observation : on ne mesure que la glycémie (premier composant)
        self.H  = np.array([[1, 0]])

        self.Q  = Q_PROFILE   # Bruit de processus (incertitude du modèle)
        self.R  = R_PROFILE   # Bruit de mesure (incertitude du capteur CGM)

        # État du gel Kalman
        self.freeze = False           # True = filtre gelé (prédiction libre sans correction)

        self._residual_window = []    # Fenêtre glissante des résidus positifs pour le CUSUM
        self._cusum_score = 0.0       # Score CUSUM courant
        self._confirm_count = 0       # Compteur de steps consécutifs au-dessus de CUSUM_H
        self._confirmed = False       # État de l'alerte confirmée
        self._neg_slope_count = 0     # Compteur de steps descendants consécutifs (pour dégel)
        self._cgm_history = [float(first_cgm)]  # Historique CGM pour calcul de pente
        self._prev_cgm = float(first_cgm)       # Valeur CGM du step précédent

    def _slope_cgm(self):
        """
        Calcule la pente de la glycémie sur les SLOPE_LOOKBACK_STEPS derniers steps.
        Utilisée uniquement pour le dégel interne.
        Retourne 0 si pas assez d'historique.
        """
        if len(self._cgm_history) <= self.SLOPE_LOOKBACK_STEPS:
            return 0.0
        # Pente = (CGM_actuel - CGM_il_y_a_N_steps) / (N * dt)
        return (
            self._cgm_history[-1] - self._cgm_history[-1 - self.SLOPE_LOOKBACK_STEPS]
        ) / (self.SLOPE_LOOKBACK_STEPS * self.dt)

    def _kalman_correction(self, x_pred, P_pred, z):
        """
        Étape de correction du filtre de Kalman (mise à jour Bayésienne).

        S = H * P_pred * H^T + R          → covariance de l'innovation
        K = P_pred * H^T * S^{-1}         → gain de Kalman
        x_new = x_pred + K * (z - H * x_pred)  → état corrigé
        P_new = (I - K * H) * P_pred      → covariance corrigée

        Paramètres :
            x_pred : état prédit (2×1)
            P_pred : covariance prédite (2×2)
            z      : mesure CGM (1×1)

        Retourne (x_new, P_new).
        """
        S = self.H @ P_pred @ self.H.T + self.R          # Covariance innovation
        K = P_pred @ self.H.T @ np.linalg.inv(S)         # Gain de Kalman
        x_new = x_pred + K @ (z - self.H @ x_pred)       # Correction de l'état
        P_new = (np.eye(2) - K @ self.H) @ P_pred        # Correction de la covariance
        return x_new, P_new

    def update(self, cgm):
        """
        Met à jour le détecteur Kalman-CUSUM avec une nouvelle mesure CGM.

        Retourne un tuple (p_libre, ecart_libre, cusum_score, confirmed) :
            p_libre      : prédiction Kalman libre (sans correction)
            ecart_libre  : résidu CGM - prédiction libre
            cusum_score  : score CUSUM courant
            confirmed    : True si l'alerte est confirmée
        """
        cgm = float(cgm)
        self._cgm_history.append(cgm)

        z = np.array([[cgm]])  # Vecteur de mesure

        # ── Étape 1 : Prédiction libre ────────────────────────────────────────
        x_pred   = self.F @ self.x           # État prédit : x_{t|t-1}
        P_pred   = self.F @ self.P @ self.F.T + self.Q  # Covariance prédite
        p_libre  = float((self.H @ x_pred)[0, 0])  # Glycémie prédite (scalaire)
        ecart_libre  = float(cgm - p_libre)   # Résidu : écart entre mesure et prédiction
        residual_pos = max(0.0, ecart_libre)  # On ne garde que les résidus positifs (CGM > prédiction)

        # ── Étape 2 : Détection de descente CGM pour le dégel ─────────────────
        delta_cgm = cgm - self._prev_cgm     # Variation depuis le dernier step
        if delta_cgm < 0:
            self._neg_slope_count += 1        # CGM baisse → incrémentation du compteur de descente
        else:
            self._neg_slope_count = 0         # CGM monte ou stable → remise à zéro

        # ── Dégel automatique : si le CGM baisse pendant N_DEGEL steps ────────
        just_degel = False
        if self.freeze and self._neg_slope_count >= self.N_DEGEL:
            # Conditions de dégel atteintes : on remet le filtre en mode normal
            self.freeze = False
            self._residual_window = []
            self._cusum_score = 0.0
            self._confirm_count = 0
            self._confirmed = False
            self._neg_slope_count = 0
            just_degel = True

            # On fait une correction Kalman normale au step du dégel
            self.x, self.P = self._kalman_correction(x_pred, P_pred, z)
            self._prev_cgm = cgm
            return p_libre, ecart_libre, 0.0, False  # Pas d'alerte au moment du dégel

        # ── Étape 3 : Activation du gel ───────────────────────────────────────
        # Le filtre est gelé dès que le CGM dépasse la prédiction libre (écart > 0)
        # Cela permet de ne pas "absorber" la dérive dans l'état du filtre
        if (ecart_libre > 0.0) and (not self.freeze):
            self.freeze = True
            self._neg_slope_count = 0  # Remise à zéro du compteur de descente

        # ── Étape 4 : Mise à jour CUSUM (uniquement en mode gel) ──────────────
        if self.freeze:
            # Accumulation du résidu positif dans la fenêtre glissante
            self._residual_window.append(residual_pos)
            if len(self._residual_window) > CUSUM_RES_WINDOW:
                self._residual_window.pop(0)  # Suppression du plus ancien résidu

            # Lissage : on utilise la moyenne de la fenêtre pour réduire le bruit
            residual_filtre = float(np.mean(self._residual_window)) if self._residual_window else 0.0

            # Mise à jour CUSUM positif : S_t = max(0, S_{t-1} + r_t - k)
            # k = CUSUM_K est le seuil de référence (bruit attendu)
            self._cusum_score = max(0.0, self._cusum_score + residual_filtre - CUSUM_K)

            # En mode gel : pas de correction Kalman → la prédiction libre continue
            self.x = x_pred
            self.P = P_pred
        else:
            # Mode normal : on réinitialise la fenêtre et le CUSUM
            self._residual_window = []
            self._cusum_score = 0.0
            # Correction Kalman standard avec la mesure CGM
            self.x, self.P = self._kalman_correction(x_pred, P_pred, z)

        # ── Étape 5 : Alarme CUSUM avec filtre de confirmation ────────────────
        slope = self._slope_cgm()

        # L'alerte est uniquement basée sur le dépassement du seuil CUSUM.
        # La pente n'entre PAS dans la décision d'alarme Kalman
        # (elle sert seulement au dégel interne).
        raw_alarm = (self._cusum_score > CUSUM_H)

        # Pas d'alarme au step du dégel ou en mode normal
        if just_degel or (not self.freeze):
            raw_alarm = False

        # Comptage des steps consécutifs au-dessus du seuil
        if raw_alarm:
            self._confirm_count += 1
        else:
            self._confirm_count = 0
            self._confirmed = False

        # Confirmation définitive après KALMAN_CONFIRM_STEPS steps consécutifs
        if self._confirm_count >= KALMAN_CONFIRM_STEPS:
            self._confirmed = True

        self._prev_cgm = cgm
        return p_libre, ecart_libre, self._cusum_score, self._confirmed

    def set_freeze(self, freeze):
        """
        Interface passive : la fusion ne commande PAS le gel/dégel de Kalman.
        Cette méthode existe pour compatibilité mais ne fait rien.
        """
        pass

    def reset(self, cgm):
        """
        Réinitialise complètement le filtre de Kalman à partir d'une nouvelle mesure CGM.
        Utilisé lors du reset de l'environnement simglucose.
        """
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


# ============================================================
# SECTION 10 — MACHINE À ÉTATS DE FUSION
# ============================================================
# Les 4 cas possibles de combinaison Kalman × LISA :

CAS_KK_LL = "K_alerte  + L_alerte  → PANNE CONFIRMÉE"        # Les deux alertent → PANNE
CAS_KK_LS = "K_alerte  + L_silence → SUSPICION Kalman"       # Seulement Kalman → SUSPICION
CAS_KS_LL = "K_silence + L_alerte  → SUSPICION LISA"         # Seulement LISA → SUSPICION
CAS_KS_LS = "K_silence + L_silence → OK"                     # Aucun → OK
CAS_SEQ   = "Alertes successives K→L ou L→K → SUSPICION SÉQUENTIELLE"  # Alertes décalées


class FusionStateMachine:
    """
    Machine à états qui combine les décisions de Kalman et LISA.

    États possibles :
        "OK"                      → pas d'anomalie détectée
        "SUSPICION"               → un seul détecteur alerte
        "SUSPICION_SEQUENTIELLE"  → les deux alertent successivement (fenêtre inter-bolus)
        "PANNE"                   → les deux alertent simultanément

    La fusion est passive : elle n'envoie aucune commande aux détecteurs.
    """

    def __init__(self):
        self.etat      = "OK"    # État courant de la fusion
        self.confiance = None    # Niveau de confiance ("HAUTE", "SEQUENTIELLE", ou None)

        # Commandes issues de la fusion vers les détecteurs (toujours False : fusion passive)
        self.cmd_freeze = False
        self.cmd_reset  = False

        # Compteurs statistiques (pour affichage et évaluation)
        self.nb_panne         = 0    # Nombre de steps en état PANNE
        self.nb_suspicion     = 0    # Nombre de steps en état SUSPICION
        self.nb_suspicion_seq = 0    # Nombre de steps en état SUSPICION_SEQUENTIELLE
        self.nb_ok            = 0    # Nombre de steps en état OK
        self.nb_panne_haute   = 0    # Nombre de steps PANNE avec confiance HAUTE

        # Compteur par cas de fusion
        self.nb_cas = {
            CAS_KK_LL: 0,
            CAS_KK_LS: 0,
            CAS_KS_LL: 0,
            CAS_KS_LS: 0,
            CAS_SEQ:   0,
        }

        # Steps auxquels chaque détecteur a alerté pour la première fois (dans l'épisode courant)
        self._k_first_alert_step = None
        self._l_first_alert_step = None

        # Historique des écarts K→L enregistrés
        self.ecarts = []

        # Mémoire du dernier détecteur seul qui a alerté (pour détection séquentielle)
        self._last_single_detector = None   # "K" ou "L"
        self._last_single_step     = None   # Step de la dernière alerte simple
        self._last_single_window   = None   # Fenêtre inter-bolus de la dernière alerte simple

    def _single_detector_courant(self, k_alert, l_alert):
        """
        Retourne "K" si seul Kalman alerte, "L" si seul LISA alerte, None sinon.
        """
        if k_alert and not l_alert:
            return "K"
        if l_alert and not k_alert:
            return "L"
        return None

    def _est_sequentielle_interbolus(self, detector_now, step, bolus_window_start):
        """
        Détecte si l'alerte actuelle est une alerte séquentielle inter-bolus :
        - Les deux détecteurs ont alerté dans la même fenêtre inter-bolus
        - Mais à des moments différents (pas simultanément)
        - L'écart est strictement positif et ≤ SEQ_WINDOW_STEPS

        Retourne True si toutes les conditions sont réunies.
        """
        if detector_now not in ("K", "L"):
            return False
        if self._last_single_detector not in ("K", "L"):
            return False
        if detector_now == self._last_single_detector:
            return False  # Même détecteur qui alerte à nouveau → pas séquentiel
        if bolus_window_start is None or self._last_single_window is None:
            return False
        if bolus_window_start != self._last_single_window:
            return False  # Fenêtre inter-bolus différente → pas le même épisode
        # Écart temporel strictement positif et dans la fenêtre autorisée
        return 0 < (step - self._last_single_step) <= SEQ_WINDOW_STEPS

    def update(self, k_alert, l_alert, step, cgm_val, sample_time, bolus_window_start=None):
        """
        Met à jour la machine à états avec les alertes courantes des deux détecteurs.

        Paramètres :
            k_alert           : True si Kalman émet une alerte confirmée
            l_alert           : True si LISA émet une alerte confirmée
            step              : numéro de step courant
            cgm_val           : valeur CGM courante (mg/dL)
            sample_time       : durée d'un step (minutes)
            bolus_window_start: step du dernier bolus (pour détection séquentielle)

        Retourne (etat, confiance, cas).
        """
        # Réinitialisation des commandes (fusion passive = jamais de commande)
        self.cmd_freeze = False
        self.cmd_reset  = False

        # Identification du détecteur seul actif (si un seul alerte)
        detector_now = self._single_detector_courant(k_alert, l_alert)

        if k_alert and l_alert:
            # ── CAS 1 : PANNE CONFIRMÉE — les deux alertent simultanément ─────
            cas = CAS_KK_LL
            self.etat = "PANNE"
            self.confiance = "HAUTE"

            # Mémorisation du premier step d'alerte de chaque détecteur
            if self._k_first_alert_step is None:
                self._k_first_alert_step = step
            if self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._enregistrer_ecart(step, "HAUTE")

            # En PANNE, on réinitialise la mémoire des alertes simples
            self._last_single_detector = None
            self._last_single_step     = None
            self._last_single_window   = None

        elif detector_now and self._est_sequentielle_interbolus(detector_now, step, bolus_window_start):
            # ── CAS 2 : SUSPICION SÉQUENTIELLE — alertes décalées dans la même fenêtre ──
            cas = CAS_SEQ
            self.etat = "SUSPICION_SEQUENTIELLE"
            self.confiance = "SEQUENTIELLE"

            # Mémorisation du step d'alerte pour le détecteur actif
            if detector_now == "K" and self._k_first_alert_step is None:
                self._k_first_alert_step = step
            if detector_now == "L" and self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._enregistrer_ecart(step, "SEQUENTIELLE")

            # On met à jour la mémoire de la dernière alerte simple
            self._last_single_detector = detector_now
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        elif k_alert and not l_alert:
            # ── CAS 3 : SUSPICION KALMAN seul ────────────────────────────────
            cas = CAS_KK_LS
            self.etat = "SUSPICION"
            self.confiance = None

            if self._k_first_alert_step is None:
                self._k_first_alert_step = step

            # Mémorisation pour détection séquentielle future
            self._last_single_detector = "K"
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        elif l_alert and not k_alert:
            # ── CAS 4 : SUSPICION LISA seule ─────────────────────────────────
            cas = CAS_KS_LL
            self.etat = "SUSPICION"
            self.confiance = None

            if self._l_first_alert_step is None:
                self._l_first_alert_step = step

            self._last_single_detector = "L"
            self._last_single_step     = step
            self._last_single_window   = bolus_window_start

        else:
            # ── CAS 5 : OK — aucun détecteur n'alerte ────────────────────────
            cas = CAS_KS_LS
            self.etat = "SUSPICION"
            self.confiance = None
            self._reset_first_alerts()  # Réinitialisation des premiers steps d'alerte

        # Incrémentation du compteur du cas courant
        self.nb_cas[cas] += 1

        # Mise à jour des compteurs statistiques selon l'état
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
        """
        Enregistre l'écart (en steps) entre les premières alertes de Kalman et LISA.
        Permet d'analyser a posteriori quel détecteur a alerté en premier.
        """
        if (self._k_first_alert_step is not None and
                self._l_first_alert_step is not None):
            # Les deux ont alerté → on peut calculer l'écart
            ecart = self._l_first_alert_step - self._k_first_alert_step
            # écart > 0 → Kalman a alerté avant LISA
            # écart < 0 → LISA a alerté avant Kalman
            self.ecarts.append((step, ecart, confiance))
        elif self._k_first_alert_step is not None:
            self.ecarts.append((step, None, confiance))   # LISA n'a pas encore alerté
        elif self._l_first_alert_step is not None:
            self.ecarts.append((step, None, confiance))   # Kalman n'a pas encore alerté

    def _reset_first_alerts(self):
        """Réinitialise les premiers steps d'alerte (retour à l'état OK)."""
        self._k_first_alert_step = None
        self._l_first_alert_step = None


# ============================================================
# SECTION 11 — INITIALISATION PATIENT & ENVIRONNEMENT
# ============================================================

# Création d'un environnement temporaire juste pour récupérer sample_time
patient    = T1DPatient.withName(PATIENT_NAME)          # Chargement du profil patient adulte#001
sensor     = CGMSensor.withName("Dexcom", seed=1)       # Capteur CGM Dexcom (seed=1 pour reproductibilité)
pump       = InsulinPump.withName("Insulet")             # Pompe à insuline Insulet
start_time = datetime.now()                              # Heure réelle de démarrage

# Environnement temporaire avec un scénario vide (juste pour lire sample_time)
env_tmp     = T1DSimEnv(patient=patient, sensor=sensor, pump=pump,
                        scenario=AdultScenario(start_time, [], 3))
sample_time = env_tmp.sample_time    # Récupération de l'intervalle CGM (en minutes, souvent 3)
del env_tmp                          # Suppression de l'environnement temporaire

# Génération du planning complet des repas sur 10 000 steps
REPAS    = build_repas_for_horizon(sample_time, horizon_steps=10000)

# Ré-instanciation propre du patient, du capteur et de la pompe (état neuf)
patient  = T1DPatient.withName(PATIENT_NAME)
sensor   = CGMSensor.withName("Dexcom", seed=1)
pump     = InsulinPump.withName("Insulet")
scenario = AdultScenario(start_time, REPAS, sample_time)  # Scénario avec les repas

# Création de l'environnement de simulation définitif
env      = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

# Ajout du champ "step" à chaque repas (pour référencement rapide par step)
for r in REPAS:
    r["step"] = int(r["min"] / sample_time)

# Dictionnaire {step_bolus → carbs} : le bolus est anticipé PREBOLUS_STEPS avant le repas
BOLUS_CARBS_STEPS = {max(0, r["step"] - PREBOLUS_STEPS): r["carbs"] for r in REPAS}
BOLUS_STEPS       = BOLUS_CARBS_STEPS  # Alias (référence partagée)


def get_inter_bolus_window_start(step, bolus_steps_dict):
    """
    Retourne le step du dernier bolus passé (la "fenêtre inter-bolus" courante).
    Utilise une recherche dichotomique pour trouver le plus grand step_bolus ≤ step.

    Paramètres :
        step             : step courant
        bolus_steps_dict : dictionnaire {step_bolus → carbs}

    Retourne :
        Le step du dernier bolus passé, ou None si aucun bolus n'est encore passé.
    """
    bs = sorted(bolus_steps_dict.keys())   # Liste triée des steps de bolus
    if not bs or step < bs[0]:
        return None  # Pas encore de bolus
    lo, hi, idx = 0, len(bs) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if bs[mid] <= step:
            idx = mid; lo = mid + 1   # On cherche le plus grand step ≤ step courant
        else:
            hi  = mid - 1
    return bs[idx]  # Dernier step de bolus passé


def update_bolus_steps(current_step, scenario, sample_time):
    """
    Met à jour le dictionnaire BOLUS_CARBS_STEPS avec les nouveaux repas
    si le scénario a été étendu.
    Appelé à chaque step pour gérer la simulation infinie.
    """
    scenario.extend_if_needed(current_step, sample_time, horizon=500)
    for r in scenario.repas:
        s_r = r.get("step")
        if s_r is None: continue
        bolus_step = max(0, s_r - PREBOLUS_STEPS)   # Step du pré-bolus correspondant
        if bolus_step not in BOLUS_CARBS_STEPS:
            BOLUS_CARBS_STEPS[bolus_step] = r["carbs"]  # Ajout du nouveau bolus prévu


MAX_SIM_STEPS = None   # Simulation continue sans limite de steps

# Affichage de la configuration au démarrage
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
print("  Durée évaluée    : simulation continue (arrêt avec Ctrl+C)")
print("  Arrêt            : Ctrl+C")
print(f"{'='*72}\n")

# En-tête du tableau de la console (affiché à chaque step)
print(f"{'Step':>5} {'Heure':>6} {'CGM':>7} "
      f"{'K_alr':>7} {'L_alr':>7} {'Score':>8} {'Consec':>7} "
      f"{'GFM':>10} {'IFM':>7} {'Pente':>7} "
      f"{'Fusion':>22} {'Confiance':>14} {'Détecteur':>15} "
      f"{'Cas':>42} {'Réel':>8}")
print("─" * 170)


# ============================================================
# SECTION 12 — CONFIGURATION DU GRAPHIQUE (4 panneaux)
# ============================================================

plt.style.use("default")  # Style matplotlib par défaut (fond blanc)

# Palette de couleurs — thème vert-bleu (pour distinguer du profil enfant en bleu)
C_BG        = "white"       # Fond des panneaux
C_GRID      = "#cccccc"     # Grille
C_SPINE     = "#aaaaaa"     # Axes
C_TICK      = "#333333"     # Labels des axes
C_CGM       = "#005f5f"     # Courbe CGM (vert-bleu foncé)
C_PRED      = "#b35900"     # Courbe prédiction Kalman (orange foncé)
C_KSCORE    = "#8b0000"     # Score CUSUM (rouge foncé)
C_GFM       = "#4b0082"     # Indicateur GFM (indigo)
C_SLOPE     = "#005f5f"     # Pente (même couleur que CGM)
C_IFM       = "#7a3500"     # Indicateur IFM (marron)
C_OR_BRUT   = "#aaaaaa"     # Compteur de steps consécutifs (gris)
C_SEUIL_K   = "#cc0000"     # Seuil CUSUM (rouge)
C_SEUIL_GFM = "#4b0082"     # Seuil GFM
C_SEUIL_SLP = "#005f5f"     # Seuil pente
C_SEUIL_IFM = "#7a3500"     # Seuil IFM
C_MEAL      = "#cc5500"     # Marqueurs repas (orange)
C_BOLUS_OK  = "#003f8a"     # Marqueurs bolus réussi (bleu)
C_BOLUS_FAIL= "#cc0000"     # Marqueurs bolus manqué (rouge)
C_HAUTE     = "#cc0000"     # Alerte PANNE (rouge)
C_SUSPICION = "#005f5f"     # Alerte SUSPICION (vert-bleu)
C_SEQ       = "#b87700"     # Alerte SÉQUENTIELLE (ambre)
C_BAND_FAIL = "#ffcccc"     # Bande de fond pompe HS (rose clair)
C_BAND_OK   = "#e6f7ee"     # Bande de fond pompe OK (vert très clair)
C_ZONE_CIBLE= "#e8f4fd"     # Zone cible glycémique [70-180 mg/dL] (bleu très clair)

# Création de la figure avec 4 panneaux superposés
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(C_BG)   # Fond de la figure

# GridSpec : 4 lignes, hauteurs [3, 1, 1, 1] (panneau CGM = 3× plus grand)
gs = gridspec.GridSpec(4, 1, figure=fig,
                       height_ratios=[3, 1, 1, 1], hspace=0.08)

# Création des sous-axes (panneaux) avec axe X partagé
ax_cgm = fig.add_subplot(gs[0])                      # Panneau principal : glycémie
ax_k   = fig.add_subplot(gs[1], sharex=ax_cgm)       # Score CUSUM (Kalman)
ax_l   = fig.add_subplot(gs[2], sharex=ax_cgm)       # Indicateurs LISA
ax_fus = fig.add_subplot(gs[3], sharex=ax_cgm)       # Décision de fusion
AXES   = [ax_cgm, ax_k, ax_l, ax_fus]               # Liste pour les opérations groupées

# Stylisation commune à tous les panneaux
for ax in AXES:
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)
    ax.tick_params(colors=C_TICK, labelsize=7)
    for s in ["top", "right"]:    ax.spines[s].set_visible(False)   # Suppression des bordures
    for s in ["bottom", "left"]:  ax.spines[s].set_color(C_SPINE)

# ── Panneau CGM (ax_cgm) ───────────────────────────────────────────────────

# Courbe CGM mesurée
line_cgm,  = ax_cgm.plot([], [], color=C_CGM,  lw=2.5, zorder=3)
# Courbe de prédiction Kalman (tirets)
line_pred, = ax_cgm.plot([], [], color=C_PRED, ls="--", lw=1.5, zorder=3, alpha=0.85)

# Marqueurs scatter (initialisés vides, remplis à chaque refresh)
sc_meals       = ax_cgm.scatter([], [], marker="^", s=160, color=C_MEAL,       zorder=6)  # Repas (triangle)
sc_bolus_ok    = ax_cgm.scatter([], [], marker="o", s=120, color=C_BOLUS_OK,   zorder=6)  # Bolus réussi (cercle)
sc_bolus_fail  = ax_cgm.scatter([], [], marker="x", s=160, color=C_BOLUS_FAIL, zorder=6, lw=2.5)  # Bolus manqué (croix)
sc_panne_haute = ax_cgm.scatter([], [], marker="D", s=180, color=C_HAUTE,      zorder=9, lw=1.5)  # PANNE (losange)
sc_suspicion   = ax_cgm.scatter([], [], marker="o", s=80,  color=C_SUSPICION,  zorder=8,          # SUSPICION (cercle vide)
                                facecolors="none", lw=2.0)
sc_susp_seq    = ax_cgm.scatter([], [], marker="s", s=100, color=C_SEQ,        zorder=8,          # SÉQUENTIELLE (carré vide)
                                facecolors="none", lw=2.0)

# Lignes horizontales : seuils d'hypo (70) et d'hyper (180)
ax_cgm.axhline(70,  color=C_BOLUS_OK,   ls="--", lw=1.2, alpha=0.7)
ax_cgm.axhline(180, color=C_BOLUS_FAIL, ls="--", lw=1.2, alpha=0.7)
# Zone cible [70-180 mg/dL] colorée en bleu très clair
ax_cgm.axhspan(70, 180, alpha=0.08, color=C_ZONE_CIBLE)
ax_cgm.set_ylim(40, 450)       # Limites de l'axe Y (glycémie en mg/dL)
ax_cgm.set_ylabel("Glycémie (mg/dL)", color=C_TICK, fontsize=8)

# Légende du panneau CGM
legend_elems = [
    Line2D([0],[0], color=C_CGM,   lw=2.5,           label="CGM"),
    Line2D([0],[0], color=C_PRED,  lw=1.5, ls="--",   label="Prédiction Kalman"),
    Patch(facecolor=C_BAND_FAIL, alpha=0.5,            label="Pompe HS (réel)"),
    Patch(facecolor=C_BAND_OK,   alpha=0.5,            label="Pompe OK (réel)"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor=C_MEAL,     ms=8, ls="None", label="Repas"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BOLUS_OK, ms=7, ls="None", label="Bolus injecté"),
    Line2D([0],[0], marker="x", color=C_BOLUS_FAIL, ms=8, ls="None",   label="Bolus manqué"),
    Line2D([0],[0], marker="D", color=C_HAUTE,      ms=8, ls="None",   label="PANNE CONFIRMÉE (K+L)"),
    Line2D([0],[0], marker="o", color=C_SUSPICION,  ms=8, ls="None",
           markerfacecolor="none", label="SUSPICION (1 détecteur)"),
    Line2D([0],[0], marker="s", color=C_SEQ,        ms=8, ls="None",
           markerfacecolor="none", label="SUSPICION SÉQUENTIELLE"),
]
ax_cgm.legend(handles=legend_elems, loc="upper left",
              fontsize=7, facecolor="white", edgecolor=C_SPINE, ncol=3)

# ── Panneau Kalman (ax_k) ──────────────────────────────────────────────────

line_kscore, = ax_k.plot([], [], color=C_KSCORE, lw=2.0)  # Score CUSUM
ax_k.axhline(CUSUM_H, color=C_SEUIL_K, ls="--", lw=1.2, alpha=0.85)  # Seuil d'alarme
ax_k.set_ylim(0, max(100, CUSUM_H * 1.4))   # Y max = 140% du seuil CUSUM_H
ax_k.set_ylabel("Score\nCUSUM", color=C_TICK, fontsize=7)

# ── Panneau LISA (ax_l) ────────────────────────────────────────────────────

line_gfm,    = ax_l.plot([], [], color=C_GFM,    lw=2.0,        label="GFM/100")  # GFM divisé par 100 pour l'échelle
line_ifm,    = ax_l.plot([], [], color=C_IFM,    lw=1.5, ls="-.", label="IFM")
line_slope,  = ax_l.plot([], [], color=C_SLOPE,  lw=1.5,         label="Pente")
line_consec, = ax_l.plot([], [], color=C_OR_BRUT, lw=1.0, ls=":",
                          label=f"Consécutifs/{LISA_CONFIRMATION_STEPS}")  # Compteur normalisé

# Lignes de seuils pour chaque indicateur LISA
ax_l.axhline(LISA_GFM_THRESHOLD / 100, color=C_SEUIL_GFM, ls="--", lw=1.2, alpha=0.85)
ax_l.axhline(LISA_IFM_THRESHOLD,        color=C_SEUIL_IFM, ls="--", lw=1.2, alpha=0.85)
ax_l.axhline(LISA_SLOPE_THRESHOLD,      color=C_SEUIL_SLP, ls="--", lw=1.2, alpha=0.85)
ax_l.axhline(1.0, color=C_OR_BRUT, ls=":", lw=1.0, alpha=0.6, label="Seuil confirmation")
ax_l.set_ylim(-2, 12)
ax_l.set_ylabel("Indicateurs\nLISA", color=C_TICK, fontsize=7)
ax_l.legend(loc="upper left", fontsize=6, facecolor="white", edgecolor=C_SPINE, ncol=2)

# ── Panneau fusion (ax_fus) ────────────────────────────────────────────────
# Échelle Y : 0=OK, 1=SUSPICION, 1.5=SÉQUENTIELLE, 2=PANNE détectée, 3=PANNE réelle
ax_fus.set_ylim(-0.5, 3.5)
ax_fus.set_yticks([0, 1, 1.5, 2, 3])
ax_fus.set_yticklabels(["OK", "SUSP.", "SUSP.\nSÉQ.", "PANNE\nfusion", "PANNE\nréelle"],
                        color=C_TICK, fontsize=6)
ax_fus.set_ylabel("Décision", color=C_TICK, fontsize=7)
ax_fus.set_xlabel("Heure simulée", color=C_TICK, fontsize=8)

# Historiques pour le panneau fusion
fusion_level_history = []  # Niveau de fusion (0/1/1.5/2) à chaque step
real_level_history   = []  # Niveau réel de panne (3 si panne, 2.85 sinon)

# ── Textes dynamiques superposés sur le panneau CGM ────────────────────────

# Glycémie courante + état de fusion (coin haut-droite)
txt_etat  = ax_cgm.text(0.99, 0.97, "", transform=ax_cgm.transAxes,
                         color=C_CGM, fontsize=13, ha="right", va="top",
                         fontfamily="monospace", fontweight="bold")
# Informations détaillées (step, alertes, indicateurs)
txt_info  = ax_cgm.text(0.99, 0.89, "", transform=ax_cgm.transAxes,
                         color="#444444", fontsize=8, ha="right", va="top")
# Cas de fusion courant + écart K→L
txt_cas   = ax_cgm.text(0.99, 0.82, "", transform=ax_cgm.transAxes,
                         color="#7a5c00", fontsize=8, ha="right", va="top",
                         fontfamily="monospace")
# Alarme principale (bandeau centré, visible en cas de PANNE)
txt_alarm = ax_cgm.text(0.5, 0.50, "", transform=ax_cgm.transAxes,
                         color=C_HAUTE, fontsize=12, ha="center", va="center",
                         fontfamily="monospace", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe5e5",
                                   edgecolor=C_HAUTE, alpha=0.0))

# Titre de la figure avec les paramètres clés
fig.suptitle(
    f"Fusion Kalman + LISA v1 — Détection de panne pompe ({PATIENT_NAME})\n"
    f"LISA: score≥{LISA_SCORE_MIN} + confirmation {LISA_CONFIRMATION_STEPS} steps "
    f"({LISA_CONFIRMATION_STEPS*3}min) | Kalman-CUSUM: k={CUSUM_K:.1f}, h={CUSUM_H:.1f}  —  Ctrl+C",
    fontsize=10, color="#111111", fontfamily="monospace"
)

# Dictionnaire pour mémoriser les bandes colorées de fond (panne/marche)
# Elles sont supprimées et recréées à chaque refresh
failure_bands = {ax: [] for ax in AXES}
fill_ref      = [None]   # Référence à la zone colorée sous la courbe CGM (liste pour mutabilité)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # Ajustement des marges (laisse de la place pour le suptitle)
plt.ion()   # Mode interactif : la fenêtre reste ouverte pendant la simulation
plt.show()  # Affichage initial de la fenêtre vide


# ============================================================
# SECTION 13 — INITIALISATION DE LA BOUCLE PRINCIPALE
# ============================================================

s    = env.reset()     # Reset de l'environnement → renvoie l'état initial (observation + done)
obs  = s.observation   # Observation initiale (contient obs.CGM : glycémie initiale)
done = s.done          # False au démarrage

index      = 0          # Compteur de steps
prev_state = None       # État précédent de la pompe (pour détecter les transitions)

# Instanciation des trois composants principaux
kalman = KalmanDetector(float(obs.CGM), sample_time)  # Détecteur Kalman initialisé avec la première mesure CGM
lisa   = LISADetector(sample_time)                    # Détecteur LISA (historique vide)
fusion = FusionStateMachine()                          # Machine à états de fusion (état "OK")

# Historiques pour le graphique (toutes les valeurs enregistrées depuis le début)
times             = []   # Labels d'heure (HH:MM) pour l'axe X
cgm_vals          = []   # Mesures CGM brutes
pred_vals         = []   # Prédictions Kalman
pump_states       = []   # État réel de la pompe (True=panne, False=OK)
k_scores          = []   # Score CUSUM à chaque step
k_alerts_h        = []   # Alertes Kalman confirmées (booléens)
l_gfm_h           = []   # GFM à chaque step
l_ifm_h           = []   # IFM à chaque step
l_slope_h         = []   # Pente CGM à chaque step
l_consec_h        = []   # Compteur de steps consécutifs LISA
l_or_brut_h       = []   # Score brut LISA (booléens)
l_alerts_h        = []   # Alertes LISA confirmées (booléens)
fusion_states     = []   # État de fusion à chaque step
cas_history       = []   # Cas de fusion à chaque step
confiance_history = []   # Niveau de confiance à chaque step

# Nombre maximum de points affichés simultanément (fenêtre glissante du graphique)
MAX_PTS = 80

# Compteurs de suivi insuline
bolus_total_prevu   = 0    # Nombre de bolus qui auraient dû être injectés
bolus_total_injecte = 0    # Nombre de bolus effectivement injectés
bolus_total_manque  = 0    # Nombre de bolus manqués à cause de la panne
insuline_injectee_U = 0.0  # Insuline totale injectée (unités)
insuline_manquee_U  = 0.0  # Insuline totale manquée à cause des pannes (unités)
insuline_basale_U   = 0.0  # Insuline basale totale injectée (unités)


# ============================================================
# SECTION 14 — BOUCLE PRINCIPALE DE SIMULATION
# ============================================================

STOP_REASON = "Ctrl+C"   # Cause d'arrêt par défaut

try:
    while True:   # Boucle infinie : Ctrl+C pour arrêter

        # ── Extension dynamique du planning de repas si nécessaire ───────────
        update_bolus_steps(index, scenario, sample_time)

        # ── Lecture de la mesure CGM courante ────────────────────────────────
        cgm_val    = float(obs.CGM)                       # Glycémie en mg/dL
        sim_min    = index * sample_time                  # Minutes simulées depuis le début
        time_label = (start_time + timedelta(minutes=sim_min)).strftime("%H:%M")  # Heure affichée

        # ── Calcul du bolus planifié pour ce step ────────────────────────────
        carbs_for_bolus = BOLUS_CARBS_STEPS.get(index, 0)  # Glucides du repas associé (0 si aucun)
        bolus_plan      = calculer_bolus(carbs_for_bolus, cgm_val) if carbs_for_bolus > 0 else 0.0

        # ── État réel de la pompe à ce step ──────────────────────────────────
        failed          = pump_is_failed(index)  # True = pompe en panne

        # ── Affichage console lors des transitions panne/marche ───────────────
        if prev_state is not None and failed != prev_state:
            sep = "─" * 110
            if failed:
                print(f"\n{sep}\n  [!] PANNE POMPE RÉELLE — step {index} ({time_label})\n{sep}\n")
            else:
                print(f"\n{sep}\n  [v] POMPE RÉTABLIE — step {index} ({time_label})\n{sep}\n")
        prev_state = failed

        # ── Calcul du débit basal dynamique (adaptatif selon la glycémie) ────
        dynamic_basal = BASAL         # Débit de base
        if cgm_val > 170: dynamic_basal += 0.008   # Légère augmentation si hyperglycémie
        if cgm_val > 210: dynamic_basal += 0.015   # Augmentation plus forte
        if cgm_val < 100: dynamic_basal -= 0.008   # Réduction si glycémie basse
        if cgm_val <  85: dynamic_basal -= 0.015   # Réduction plus forte
        if cgm_val <  70: dynamic_basal = 0.0      # Arrêt total si hypoglycémie
        dynamic_basal = max(0.0, dynamic_basal)    # Toujours ≥ 0

        # ── Bolus de correction automatique si hyperglycémie sans repas ──────
        if bolus_plan == 0.0 and cgm_val > 200.0:
            # Correction pure : dose proportionnelle à l'écart par rapport à la cible
            bolus_final = min(4.0, max(0.0, (cgm_val - CGM_CIBLE) / ISF))
        else:
            bolus_final = bolus_plan   # Bolus repas classique

        # ── Construction de l'action envoyée à la pompe ──────────────────────
        # Si la pompe est en panne → action nulle (0, 0) : aucune insuline
        # Sinon → action avec le débit basal et le bolus calculés
        action = (Action(basal=0.0, bolus=0.0) if failed else
                  Action(basal=dynamic_basal, bolus=bolus_final))

        # Dose d'insuline effective reçue par le patient à ce step
        insulin_dose_step = (0.0 if failed else
                             dynamic_basal * sample_time + bolus_final)

        # ── Mise à jour du détecteur Kalman ──────────────────────────────────
        # Retourne : prédiction libre, résidu, score CUSUM, alerte confirmée
        pred, residual, k_score, k_alert = kalman.update(cgm_val)

        # ── Mise à jour du détecteur LISA ─────────────────────────────────────
        # Retourne : GFM, IFM, pente, alerte confirmée, score brut, compteur consécutif
        gfm, ifm, slope, l_alert, or_brut, consec = lisa.update(cgm_val, insulin_dose_step)

        # ── Mise à jour de la machine à états de fusion ───────────────────────
        bolus_window_start = get_inter_bolus_window_start(index, BOLUS_CARBS_STEPS)
        etat_fusion, confiance, cas = fusion.update(
            k_alert, l_alert, index, cgm_val, sample_time, bolus_window_start
        )

        # ── Identification du détecteur responsable de l'alerte ───────────────
        if k_alert and l_alert:    detecteur_txt = "Kalman + LISA"
        elif k_alert:              detecteur_txt = "Kalman seul"
        elif l_alert:              detecteur_txt = "LISA seule"
        else:                      detecteur_txt = "Aucun"

        # ── Envoi des notifications selon l'état de fusion ────────────────────
        if etat_fusion == "PANNE":
            notifier_panne(index)   # Notification urgente
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":
            # Calcul de l'écart entre les deux alertes
            _ecart_seq = 0
            if (fusion._k_first_alert_step is not None and
                    fusion._l_first_alert_step is not None):
                _ecart_seq = abs(fusion._l_first_alert_step - fusion._k_first_alert_step)
            notifier_suspicion_sequentielle(index, _ecart_seq, sample_time)
        elif etat_fusion == "SUSPICION":
            notifier_suspicion(index)   # Notification de surveillance

        # ── Fusion passive : aucune commande vers les détecteurs ───────────────
        # (commentaires explicatifs conservés ici pour documentation)
        # - Pas de kalman.reset(cgm_val) commandé par la fusion
        # - Pas de lisa.reset() commandé par la fusion
        # - Pas de kalman.set_freeze(...) commandé par la fusion

        # ── Suivi des bolus prévus vs réels ───────────────────────────────────
        bolus_attendu_ce_step = index in BOLUS_CARBS_STEPS   # True si un bolus était prévu
        bolus_manque_ce_step  = failed and bolus_attendu_ce_step  # True si prévu mais raté

        # ── Affichage console à chaque step ───────────────────────────────────
        conf_txt = confiance if confiance else "—"
        or_txt   = "OR" if or_brut else "  "   # "OR" = score brut LISA ≥ min
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

        # ── Enregistrement dans les historiques ───────────────────────────────
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

        # Conversion de l'état de fusion en valeur numérique pour le graphique
        fusion_level_history.append(
            2   if etat_fusion == "PANNE" else
            1.5 if etat_fusion == "SUSPICION_SEQUENTIELLE" else
            1   if etat_fusion == "SUSPICION" else 0
        )
        # Niveau réel : 3 = pompe en panne, 2.85 = pompe OK (légèrement en-dessous de 3)
        real_level_history.append(3 if failed else 2.85)

        # ── Suivi insuline ────────────────────────────────────────────────────
        if bolus_attendu_ce_step:
            bolus_total_prevu += 1         # Incrément du compteur de bolus prévus
            if failed:
                bolus_total_manque  += 1   # Bolus manqué (pompe HS)
                insuline_manquee_U  += bolus_plan
            else:
                bolus_total_injecte += 1   # Bolus réussi
                insuline_injectee_U += bolus_final
        elif not failed and bolus_final > 0:
            insuline_injectee_U += bolus_final  # Bolus de correction (sans repas)
        if not failed:
            insuline_basale_U += dynamic_basal * sample_time  # Insuline basale cumulée


        # ── Rafraîchissement du graphique ─────────────────────────────────────

        # Sélection des MAX_PTS derniers points pour l'affichage (fenêtre glissante)
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

        # Abscisses : indices 0..N-1 (le graphique est décalé à chaque step)
        x      = list(range(len(disp_times)))
        # Offset pour recalculer la position absolue des markers (repas, bolus, alertes)
        offset = max(0, len(times) - MAX_PTS)

        # Mise à jour des courbes principales
        line_cgm.set_data(x, disp_cgm)
        line_pred.set_data(x, disp_pred)
        line_kscore.set_data(x, disp_kscore)
        line_gfm.set_data(x, [v / 100.0 for v in disp_gfm])   # GFM divisé par 100 pour l'échelle
        line_ifm.set_data(x, disp_ifm)
        line_slope.set_data(x, disp_slope)
        # Compteur normalisé par LISA_CONFIRMATION_STEPS (pour comparer à la ligne "seuil=1")
        line_consec.set_data(x, [v / max(1, LISA_CONFIRMATION_STEPS) for v in disp_consec])

        # Zone colorée sous la courbe CGM (fill_between mis à jour à chaque step)
        if fill_ref[0] is not None:
            fill_ref[0].remove()   # Suppression de l'ancien fill
        fill_ref[0] = ax_cgm.fill_between(x, disp_cgm, alpha=0.08, color=C_CGM, zorder=2)

        # ── Panneau fusion (barres colorées) ─────────────────────────────────
        ax_fus.cla()   # Effacement complet (les barres sont recréées à chaque refresh)
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

        # Dessin de chaque barre du panneau fusion + marqueur de l'état réel
        for i_f, (fv, rv, fc) in enumerate(zip(disp_fus, disp_real, disp_conf)):
            if fv == 2:     color_f = C_HAUTE    # PANNE détectée → rouge
            elif fv == 1.5: color_f = C_SEQ      # SÉQUENTIELLE → ambre
            elif fv == 1:   color_f = "#e6a817"  # SUSPICION → jaune
            else:           color_f = "#c8e6c9"  # OK → vert clair
            # Barre verticale colorée
            ax_fus.bar(i_f, fv + 0.5, bottom=-0.5, width=1.0,
                       color=color_f, alpha=0.80, align="edge")
            # Marqueur horizontal représentant la panne réelle (pour comparaison visuelle)
            ax_fus.plot(i_f + 0.5, rv, marker="_", ms=10,
                        color=C_BOLUS_FAIL, lw=2, alpha=0.9)

        # ── Synchronisation des axes X ────────────────────────────────────────
        for ax in AXES:
            ax.set_xlim(-1, len(disp_times))

        # Ticks de l'axe X tous les 6 steps (18 min avec sample_time=3)
        ticks = list(range(0, len(disp_times), 6))
        ax_fus.set_xticks(ticks)
        ax_fus.set_xticklabels(
            [disp_times[i] for i in ticks if i < len(disp_times)],
            fontsize=7, color="#64748b"
        )
        for ax in [ax_cgm, ax_k, ax_l]:
            ax.set_xticks(ticks)
            ax.set_xticklabels([], visible=False)   # Masquer les labels X sur les 3 panneaux du haut

        # ── Bandes de fond colorées (panne/marche) ────────────────────────────
        # Suppression des anciennes bandes
        for ax in AXES:
            for band in failure_bands[ax]:
                try: band.remove()
                except Exception: pass
            failure_bands[ax].clear()

        # Création des nouvelles bandes par segments consécutifs de même état
        i_seg = 0
        while i_seg < len(disp_states):
            seg_state = disp_states[i_seg]; j_seg = i_seg
            # Extension du segment tant que l'état ne change pas
            while j_seg < len(disp_states) and disp_states[j_seg] == seg_state:
                j_seg += 1
            c_band = C_BAND_FAIL if seg_state else C_BAND_OK   # Couleur selon l'état
            a_band = 0.45 if seg_state else 0.35               # Transparence (panne plus visible)
            for ax in AXES:
                # axvspan = zone verticale colorée entre deux abscisses
                b = ax.axvspan(i_seg - 0.5, j_seg - 0.5, alpha=a_band, color=c_band, zorder=1)
                failure_bands[ax].append(b)
            i_seg = j_seg

        # ── Marqueurs des repas ───────────────────────────────────────────────
        mx, my = [], []
        for r in scenario.repas:
            r_step = r.get("step")
            if r_step is None: continue
            vi = r_step - offset   # Position dans la fenêtre affichée
            if 0 <= vi < len(disp_times):
                mx.append(vi); my.append(disp_cgm[vi] + 18)   # Légèrement au-dessus de la courbe CGM
        sc_meals.set_offsets(np.c_[mx, my] if mx else np.empty((0, 2)))

        # ── Marqueurs des bolus ───────────────────────────────────────────────
        ok_x, ok_y, fx, fy = [], [], [], []
        for step_b in BOLUS_STEPS:
            vi = step_b - offset
            if 0 <= vi < len(disp_times):
                if pump_is_failed(step_b):
                    fx.append(vi); fy.append(disp_cgm[vi] - 18)   # Bolus manqué (croix rouge, en-dessous)
                else:
                    ok_x.append(vi); ok_y.append(disp_cgm[vi] - 18)  # Bolus réussi (cercle, en-dessous)
        sc_bolus_ok.set_offsets(np.c_[ok_x, ok_y] if ok_x else np.empty((0, 2)))
        sc_bolus_fail.set_offsets(np.c_[fx, fy] if fx else np.empty((0, 2)))

        # ── Marqueurs des alertes ─────────────────────────────────────────────
        ph_x, ph_y, su_x, su_y, sq_x, sq_y = [], [], [], [], [], []
        for i_a, (fs, fc) in enumerate(zip(disp_fusion_states, disp_conf)):
            if fs == "PANNE":
                ph_x.append(i_a); ph_y.append(disp_cgm[i_a] + 30)   # Losange rouge au-dessus
            elif fs == "SUSPICION_SEQUENTIELLE":
                sq_x.append(i_a); sq_y.append(disp_cgm[i_a] + 25)   # Carré ambre
            elif fs == "SUSPICION":
                su_x.append(i_a); su_y.append(disp_cgm[i_a] + 20)   # Cercle vert-bleu
        sc_panne_haute.set_offsets(np.c_[ph_x, ph_y] if ph_x else np.empty((0, 2)))
        sc_susp_seq.set_offsets(np.c_[sq_x, sq_y] if sq_x else np.empty((0, 2)))
        sc_suspicion.set_offsets(np.c_[su_x, su_y] if su_x else np.empty((0, 2)))

        # ── Mise à jour des textes dynamiques ────────────────────────────────
        # Couleur du texte d'état selon la gravité
        if etat_fusion == "PANNE":                     col_etat = C_HAUTE
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":  col_etat = C_SEQ
        elif etat_fusion == "SUSPICION":               col_etat = "#e6a817"
        else:                                          col_etat = C_CGM

        # Texte principal : glycémie + état
        txt_etat.set_text(f"{cgm_val:.0f} mg/dL  [{etat_fusion}]")
        txt_etat.set_color(col_etat)

        # Texte d'information détaillée
        txt_info.set_text(
            f"{time_label} | step {index} | "
            f"Kalman {'✓' if k_alert else '·'}  LISA {'✓' if l_alert else '·'} "
            f"(Score≥{LISA_SCORE_MIN}: {'✓' if or_brut else '·'} ×{consec}/{LISA_CONFIRMATION_STEPS})  |  "
            f"GFM={gfm:.1f}  IFM={ifm:.3f}  Pente={slope:.2f}  |  "
            f"Réel : {'PANNE' if failed else 'OK'}"
        )

        # Texte du cas de fusion + écart K→L
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

        # ── Bandeau d'alarme central (visible seulement en état non-OK) ──────
        if etat_fusion == "PANNE":
            txt_alarm.set_text("PANNE CONFIRMÉE — Kalman + LISA")
            txt_alarm.set_color(C_HAUTE)
            txt_alarm.get_bbox_patch().set_edgecolor(C_HAUTE)
            txt_alarm.get_bbox_patch().set_facecolor("#ffe5e5")
            txt_alarm.get_bbox_patch().set_alpha(0.90)
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":
            # Calcul de l'ordre d'alerte (K avant L ou L avant K)
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
            # État OK : bandeau invisible
            txt_alarm.set_text("")
            txt_alarm.get_bbox_patch().set_alpha(0.0)

        # ── Rendu du graphique ────────────────────────────────────────────────
        try:
            fig.canvas.draw_idle()    # Redessine uniquement les éléments modifiés
            fig.canvas.flush_events() # Traitement des événements Tkinter (clics, redim.)
        except Exception:
            pass   # Ignore les erreurs graphiques (ex: fenêtre fermée par l'utilisateur)

        # ── Pause entre deux steps ────────────────────────────────────────────
        time.sleep(STEP_DELAY)   # STEP_DELAY secondes = vitesse de la simulation

        # ── Avance de la simulation ───────────────────────────────────────────
        if done:
            # L'environnement simglucose a atteint sa fin naturelle : on le recrée
            print(f"\n[INFO] Reset environnement (step {index})\n")
            p2  = T1DPatient.withName(PATIENT_NAME)
            s2  = CGMSensor.withName("Dexcom", seed=index)   # Seed différent à chaque reset
            pu2 = InsulinPump.withName("Insulet")
            sc2 = scenario
            # Mise à jour du start_time du scénario pour continuer à l'heure courante
            sc2.start_time = start_time + timedelta(minutes=index * sample_time)
            env2 = T1DSimEnv(patient=p2, sensor=s2, pump=pu2, scenario=sc2)
            s    = env2.reset()
            env  = env2
            # Réinitialisation des détecteurs avec la nouvelle mesure CGM
            kalman = KalmanDetector(float(s.observation.CGM), sample_time)
            lisa   = LISADetector(sample_time)
        else:
            # Étape normale : on avance d'un step avec l'action calculée
            s = env.step(action)

        # Mise à jour de l'observation pour le prochain step
        obs  = s.observation
        done = s.done
        index += 1    # Incrément du compteur de steps

# ============================================================
# SECTION 15 — FIN DE LA SIMULATION (Ctrl+C)
# ============================================================

except KeyboardInterrupt:
    # L'utilisateur a appuyé sur Ctrl+C : arrêt propre
    print(f"\n[INFO] Simulation arrêtée par l'utilisateur au step {index}.")

# Désactivation du mode interactif matplotlib (nécessaire pour fermeture propre)
plt.ioff()
