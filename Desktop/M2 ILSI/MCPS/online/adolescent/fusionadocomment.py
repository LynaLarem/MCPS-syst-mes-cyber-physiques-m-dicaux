import time                          # Module pour les pauses temporelles (time.sleep)
import threading                      # Module pour exécuter des tâches en parallèle (notifications asynchrones)
import urllib.request                  # Module pour envoyer des requêtes HTTP (notifications ntfy)
import numpy as np                    # Bibliothèque de calcul numérique (matrices, moyennes, etc.)
import matplotlib                     # Bibliothèque de visualisation graphique
matplotlib.use("TkAgg")               # Sélectionne le backend TkAgg pour affichage en fenêtre desktop
import matplotlib.pyplot as plt       # Interface principale pour créer des graphiques
import matplotlib.gridspec as gridspec # Outil pour organiser plusieurs panneaux dans une figure
from matplotlib.lines import Line2D   # Classe pour créer des éléments de légende sous forme de lignes
from matplotlib.patches import Patch  # Classe pour créer des éléments de légende sous forme de rectangles
from datetime import datetime, timedelta  # Classes pour manipuler les dates et durées
from collections import namedtuple    # Outil pour créer des tuples nommés (structure Action)

from simglucose.simulation.env import T1DSimEnv          # Environnement de simulation du diabète de type 1
from simglucose.simulation.scenario import Scenario       # Classe de base pour définir un scénario de repas
from simglucose.simulation.scenario import Action as MealAction  # Action repas (quantité de glucides)
from simglucose.patient.t1dpatient import T1DPatient      # Modèle physiologique du patient diabétique
from simglucose.sensor.cgm import CGMSensor               # Modèle de capteur de glucose en continu (CGM)
from simglucose.actuator.pump import InsulinPump          # Modèle de pompe à insuline

# CONFIGURATION PATIENT ADOLESCENT


PATIENT_NAME = "adolescent#001"       # Identifiant du profil patient adolescent dans simglucose

REPAS_JOURNEE = [                     # Liste des repas types dans une journée simulée (cycle de 600 min)
    {"min_in_day":  60, "carbs": 45, "label": "Petit-dej\n45g"},   # Petit-déjeuner à 60 min : 45g de glucides
    {"min_in_day": 180, "carbs": 60, "label": "Dejeuner\n60g"},    # Déjeuner à 180 min : 60g de glucides
    {"min_in_day": 300, "carbs": 20, "label": "Gouter\n20g"},      # Goûter à 300 min : 20g de glucides
    {"min_in_day": 420, "carbs": 55, "label": "Diner\n55g"},       # Dîner à 420 min : 55g de glucides
    {"min_in_day": 540, "carbs": 15, "label": "Collation\n15g"},   # Collation à 540 min : 15g de glucides
]

ICR       = 15.0    # Insulin-to-Carb Ratio : 1 unité d'insuline pour 15g de glucides
ISF       = 60.0    # Insulin Sensitivity Factor : 1 unité corrige 60 mg/dL de glycémie
CGM_CIBLE = 120.0   # Glycémie cible en mg/dL pour le calcul du bolus de correction
BOLUS_MIN = 0.3     # Dose minimale de bolus autorisée en unités
BOLUS_MAX = 10.0    # Dose maximale de bolus autorisée en unités


def calculer_bolus(carbs, cgm_val):       # Calcule la dose de bolus en fonction des glucides et de la glycémie
    if cgm_val < 70.0:                    # Si hypoglycémie (glycémie < 70 mg/dL), aucun bolus
        return 0.0                        # Retourne 0 pour éviter d'aggraver l'hypoglycémie
    bolus_repas      = carbs / ICR        # Calcule le bolus repas selon le ratio glucides/insuline
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0  # Bolus de correction si glycémie ≥ 80
    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)  # Borne et arrondit le bolus total


MINUTES_PAR_JOUR = 600   # Durée d'un cycle journalier simulé : 600 minutes (5 repas sur 10h)
REPAS            = []    # Liste globale des repas qui sera remplie lors de l'initialisation


def build_repas_for_horizon(sample_time, horizon_steps=10000):  # Génère tous les repas sur l'horizon de simulation
    repas     = []                                               # Liste vide pour stocker les repas générés
    total_min = horizon_steps * sample_time                      # Durée totale en minutes de l'horizon
    nb_jours  = int(total_min / MINUTES_PAR_JOUR) + 2           # Nombre de jours à couvrir (avec marge de 2 jours)
    for jour in range(nb_jours):                                 # Boucle sur chaque jour
        base_min = jour * MINUTES_PAR_JOUR                       # Offset temporel du début du jour courant
        for r in REPAS_JOURNEE:                                  # Boucle sur chaque repas type de la journée
            repas.append({                                       # Ajoute le repas à la liste avec ses informations
                "min"   : base_min + r["min_in_day"],            # Minute absolue du repas dans la simulation
                "carbs" : r["carbs"],                            # Quantité de glucides du repas
                "bolus" : None,                                  # Bolus calculé (initialisé à None)
                "label" : r["label"],                            # Étiquette d'affichage du repas
            })
    return repas                                                 # Retourne la liste complète des repas


PANNE_DUREE      = 32   # Durée d'une panne pompe en steps (32 steps = ~96 min pour ado)
MARCHE_DUREE     = 20   # Durée de fonctionnement normal entre pannes en steps (20 steps = ~60 min)
PANNE_START_STEP = 10   # Step à partir duquel le cycle panne/marche commence
BASAL            = 0.040  # Débit basal nominal de la pompe en U/min pour le patient adolescent
PREBOLUS_STEPS   = 8    # Nombre de steps avant le repas où le bolus est injecté (pré-bolus)
STEP_DELAY       = 1.5  # Délai en secondes entre chaque step pour ralentir la simulation visuelle
# Simulation continue : arrêt uniquement avec Ctrl+C
SIM_DURATION_HOURS = None  # Durée de simulation illimitée (None = pas de fin automatique)

Action = namedtuple("Action", ["basal", "bolus"])  # Tuple nommé représentant une action de la pompe (basal + bolus)

# PARAMÈTRES LISA — ADOLESCENT


LISA_LW                 = 20           # Taille de la grande fenêtre (Long Window) de LISA en steps
LISA_SW                 = 3            # Taille de la petite fenêtre (Short Window) de LISA en steps
LISA_GFM_THRESHOLD      = 355.379392   # Seuil du GFM (Glucose Failure Metric) pour déclencher une alerte
LISA_IFM_THRESHOLD      = 0.320989     # Seuil de l'IFM (Insulin Failure Metric) pour déclencher une alerte
LISA_SLOPE_THRESHOLD    = 0.942186     # Seuil de la pente de glycémie (mg/dL/min) pour alerte
LISA_SCORE_MIN          = 3            # Score minimum requis pour considérer une alerte LISA (sur 4 points)
LISA_PIE_ALPHA          = 0.85         # Facteur de décroissance exponentielle du PIE (insuline active estimée)
LISA_CONFIRMATION_STEPS = 3            # Nombre de steps consécutifs avec score suffisant pour confirmer une alerte


# PARAMÈTRES KALMAN — ADOLESCENT


KALMAN_WINDOW        = 20             # Fenêtre glissante en steps pour la moyenne des résidus Kalman
KALMAN_THRESHOLD     = 43.728771      # Seuil de la moyenne des résidus Kalman pour déclencher une alerte

Q_PROFILE = np.array([[1.0, 0.0],     # Matrice de bruit de processus Q : incertitude sur glycémie
                       [0.0, 0.10]])   # Incertitude sur la dérivée de la glycémie
R_PROFILE = np.array([[25.0]])        # Matrice de bruit de mesure R : variance du capteur CGM


# PARAMÈTRES MACHINE À ÉTATS FUSION

SEQ_WINDOW_STEPS = 40   # Fenêtre maximale (en steps) pour qu'une alerte séquentielle soit comptabilisée (~120 min)



# NOTIFICATIONS NTFY

NTFY_SUSPICION_URL     = "https://ntfy.sh/pompe_suspicion_detecteur"       # URL ntfy pour alerte suspicion simple
NTFY_SUSPICION_SEQ_URL = "https://ntfy.sh/pompe_suspicion_sequentielles"   # URL ntfy pour alerte suspicion séquentielle
NTFY_PANNE_URL         = "https://ntfy.sh/pompe_panne_confirme"            # URL ntfy pour alerte panne confirmée

NTFY_COOLDOWN_STEPS = 10  # Nombre de steps minimum entre deux notifications du même type (anti-spam)

_ntfy_last_suspicion     = -999  # Step du dernier envoi d'une notification suspicion (initialisé très bas)
_ntfy_last_suspicion_seq = -999  # Step du dernier envoi d'une notification suspicion séquentielle
_ntfy_last_panne         = -999  # Step du dernier envoi d'une notification panne confirmée



def _send_ntfy(url, title, message, priority="default", tags="rotating_light,warning"):  # Envoie une notification push via ntfy.sh
    def _do_send():                                              # Fonction interne exécutée dans un thread séparé
        try:
            data = message.encode("utf-8")                      # Encode le message en bytes UTF-8
            req  = urllib.request.Request(url, data=data, method="POST")  # Crée la requête HTTP POST
            req.add_header("Title",        title)               # Ajoute le titre de la notification
            req.add_header("Priority",     priority)            # Ajoute le niveau de priorité
            req.add_header("Tags",         tags)                # Ajoute les emojis/tags ntfy
            req.add_header("Content-Type", "text/plain; charset=utf-8")  # Spécifie le type de contenu
            with urllib.request.urlopen(req, timeout=8) as resp:          # Envoie la requête avec timeout 8s
                print(f"[NTFY] HTTP {resp.status} -> {url.split('/')[-1]}")  # Affiche le code HTTP de réponse
        except Exception as e:
            print(f"[NTFY] Erreur envoi notification : {e}")    # Affiche l'erreur si l'envoi échoue
    threading.Thread(target=_do_send, daemon=False).start()     # Lance l'envoi dans un thread non-daemon


def notifier_suspicion(step):                                    # Envoie une notification de suspicion simple
    global _ntfy_last_suspicion                                  # Accède à la variable globale de suivi

    if step - _ntfy_last_suspicion >= NTFY_COOLDOWN_STEPS:       # Vérifie que le cooldown est respecté
        _ntfy_last_suspicion = step                              # Met à jour le step du dernier envoi

        _send_ntfy(                                              # Envoie la notification via ntfy
            NTFY_SUSPICION_URL,                                  # URL du canal ntfy dédié
            "Suspicion pompe [ado]",                             # Titre de la notification
            "Anomalie detectee par un seul detecteur - surveillance renforcee.",  # Corps du message
            priority="high",                                     # Priorité haute
            tags="warning,mag"                                   # Emojis d'avertissement
        )

        print(f"[NTFY] Notification SUSPICION envoyee (step {step})")  # Confirme l'envoi en console


def notifier_suspicion_sequentielle(step, ecart_steps, sample_time):  # Envoie notification de suspicion séquentielle
    global _ntfy_last_suspicion_seq                              # Accède à la variable globale de suivi

    if step - _ntfy_last_suspicion_seq >= NTFY_COOLDOWN_STEPS:   # Vérifie le cooldown anti-spam
        _ntfy_last_suspicion_seq = step                          # Met à jour le step du dernier envoi

        ecart_min = abs(ecart_steps) * sample_time               # Convertit l'écart de steps en minutes

        _send_ntfy(                                              # Envoie la notification via ntfy
            NTFY_SUSPICION_SEQ_URL,                              # URL du canal dédié aux suspicions séquentielles
            "Suspicion elevee - alertes successives [ado]",      # Titre de la notification
            f"Les deux detecteurs ont alerte successivement (ecart : {ecart_min:.0f} min).",  # Message avec écart en minutes
            priority="high",                                     # Priorité haute
            tags="warning,rotating_light"                        # Emojis d'alerte
        )

        print(f"[NTFY] Notification SUSPICION SEQUENTIELLE envoyee (step {step})")  # Confirmation console


def notifier_panne(step):                                        # Envoie une notification de panne confirmée
    global _ntfy_last_panne                                      # Accède à la variable globale de suivi

    if step - _ntfy_last_panne >= NTFY_COOLDOWN_STEPS:           # Vérifie le cooldown anti-spam
        _ntfy_last_panne = step                                  # Met à jour le step du dernier envoi

        _send_ntfy(                                              # Envoie la notification via ntfy
            NTFY_PANNE_URL,                                      # URL du canal dédié aux pannes confirmées
            "PANNE CONFIRMEE [ado]",                             # Titre de la notification
            "Les deux detecteurs signalent une panne simultanee - verifiez la pompe.",  # Message urgent
            priority="urgent",                                   # Priorité maximale (urgente)
            tags="rotating_light,sos"                            # Emojis d'urgence
        )

        print(f"[NTFY] Notification PANNE CONFIRMEE envoyee (step {step})")  # Confirmation console



# UTILITAIRES SIMULATION


def pump_is_failed(step):                                        # Détermine si la pompe est en panne à un step donné
    if step < PANNE_START_STEP:                                  # Avant le début des pannes, la pompe fonctionne
        return False                                             # Retourne False (pas en panne)
    return (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE) < PANNE_DUREE  # Cycle panne/marche alterné


class AdoScenario(Scenario):                                     # Scénario de simulation spécifique à l'adolescent
    def __init__(self, start_time, repas, sample_time):          # Constructeur du scénario
        self.start_time  = start_time                            # Heure de début de la simulation
        self.repas       = repas                                 # Liste des repas prévus
        self.sample_time = sample_time                           # Intervalle de temps entre chaque step (en minutes)
        self.meal_steps  = {int(r["min"] / sample_time): r["carbs"] for r in repas}  # Dictionnaire {step: glucides}

    def get_action(self, t):                                     # Retourne l'action (repas) au temps t
        elapsed = (t - self.start_time).total_seconds() / 60    # Calcule le temps écoulé en minutes depuis le début
        step    = int(round(elapsed / self.sample_time))         # Convertit le temps en numéro de step
        return MealAction(meal=self.meal_steps.get(step, 0))     # Retourne les glucides du repas (0 si pas de repas)

    def reset(self):                                             # Méthode de réinitialisation requise par l'interface
        pass                                                     # Aucune action nécessaire au reset pour ce scénario

    def extend_if_needed(self, current_step, sample_time, horizon=500):  # Étend le planning de repas si nécessaire
        max_step = max(self.meal_steps.keys()) if self.meal_steps else 0  # Dernier step avec un repas planifié
        if current_step + horizon > max_step:                    # Si l'horizon dépasse le planning actuel
            last_min      = max_step * sample_time               # Dernière minute couverte par le planning
            nb_jours_supp = int((current_step + horizon) * sample_time / MINUTES_PAR_JOUR) + 2  # Jours supplémentaires à ajouter
            for jour in range(nb_jours_supp):                    # Boucle sur les nouveaux jours à planifier
                base_min = last_min + (jour + 1) * MINUTES_PAR_JOUR  # Offset du nouveau jour
                for r in REPAS_JOURNEE:                          # Boucle sur les repas types de la journée
                    new_min  = base_min + r["min_in_day"]        # Minute absolue du nouveau repas
                    new_step = int(new_min / sample_time)        # Step correspondant au nouveau repas
                    if new_step not in self.meal_steps:          # Évite les doublons
                        self.meal_steps[new_step] = r["carbs"]   # Ajoute le repas au dictionnaire
                        self.repas.append({                      # Ajoute le repas à la liste complète
                            "min"  : new_min,                    # Minute absolue
                            "carbs": r["carbs"],                 # Quantité de glucides
                            "bolus": None,                       # Bolus non encore calculé
                            "label": r["label"],                 # Étiquette d'affichage
                            "step" : new_step,                   # Numéro de step du repas
                        })


# DÉTECTEUR LISA — adolescent


class LISADetector:                                              # Détecteur LISA (Logic of Insulin and Sugar Assessment)
    def __init__(self, sample_time):                             # Constructeur du détecteur LISA
        self.sample_time     = sample_time                       # Intervalle de temps entre steps (en minutes)
        self.cgm_history     = []                                # Historique des mesures CGM
        self.pie_history     = []                                # Historique du PIE (insuline active estimée)
        self._pie_current    = 0.0                               # Valeur courante du PIE (initialisée à 0)
        self._gfm_current    = 0.0                               # Valeur courante du GFM (accumulateur)
        self._sw_prev        = None                              # Moyenne SW du step précédent (pour calcul de pente)
        self._or_consecutive = 0                                 # Compteur de steps consécutifs avec score suffisant
        self._confirmed      = False                             # Indicateur de confirmation de l'alerte LISA

    def update(self, cgm_val, insulin_dose):                     # Met à jour le détecteur avec une nouvelle mesure
        dt = self.sample_time                                    # Alias local pour le pas de temps

        # PIE
        self._pie_current = LISA_PIE_ALPHA * self._pie_current + insulin_dose  # Mise à jour du PIE par filtre exponentiel
        self.pie_history.append(self._pie_current)               # Enregistre la nouvelle valeur du PIE
        self.cgm_history.append(cgm_val)                         # Enregistre la nouvelle mesure CGM

        if len(self.cgm_history) < LISA_LW + 1:                 # Pas assez d'historique pour calculer les fenêtres
            self._or_consecutive = 0                             # Réinitialise le compteur de confirmations
            self._confirmed      = False                         # Aucune alerte confirmée
            return 0.0, 0.0, 0.0, False, False, 0               # Retourne des valeurs neutres

        # Fenêtres LW et SW
        cgm_arr    = np.array(self.cgm_history[-LISA_LW:])       # Tableau CGM sur la grande fenêtre LW
        pie_arr    = np.array(self.pie_history[-LISA_LW:])       # Tableau PIE sur la grande fenêtre LW
        glucose_lw = float(np.mean(cgm_arr))                     # Moyenne du CGM sur la grande fenêtre LW
        glucose_sw = float(np.mean(self.cgm_history[-LISA_SW:])) # Moyenne du CGM sur la petite fenêtre SW
        pie_lw     = float(np.mean(pie_arr))                     # Moyenne du PIE sur la grande fenêtre LW
        pie_sw     = float(np.mean(self.pie_history[-LISA_SW:])) # Moyenne du PIE sur la petite fenêtre SW

        # GFM cumulatif
        diff = glucose_sw - glucose_lw                           # Différence entre moyenne courte et longue du CGM
        if diff > 0:                                             # Si la glycémie monte (tendance haussière)
            self._gfm_current = self._gfm_current + diff * dt   # Accumule le GFM proportionnellement au temps
        else:                                                    # Si la glycémie descend
            self._gfm_current = 0.0                              # Réinitialise le GFM (pas d'accumulation en descente)
        gfm = self._gfm_current                                  # Valeur finale du GFM pour ce step

        # IFM
        ifm = max(0.0, (pie_lw - pie_sw) / pie_lw) if pie_lw > 1e-6 else 0.0  # IFM : chute relative d'insuline active (0 si indéfini)

        # Pente
        slope         = (glucose_sw - self._sw_prev) / dt if self._sw_prev is not None else 0.0  # Pente = variation de la moyenne SW
        self._sw_prev = glucose_sw                               # Sauvegarde la moyenne SW courante pour le prochain step

        # Score pondéré
        lisa_score = (2 * int(gfm   > LISA_GFM_THRESHOLD) +     # GFM compte double dans le score (poids 2)
                      1 * int(ifm   > LISA_IFM_THRESHOLD) +     # IFM compte pour 1 dans le score
                      1 * int(slope > LISA_SLOPE_THRESHOLD))     # Pente compte pour 1 dans le score

        score_signal = lisa_score >= LISA_SCORE_MIN              # True si le score dépasse le seuil minimum requis

        # Filtre de confirmation
        if score_signal:                                         # Si le score est suffisant ce step
            self._or_consecutive += 1                            # Incrémente le compteur de confirmations consécutives
        else:                                                    # Sinon
            self._or_consecutive = 0                             # Remet le compteur à zéro
            self._confirmed      = False                         # Retire la confirmation de l'alerte

        if self._or_consecutive >= LISA_CONFIRMATION_STEPS:      # Si assez de steps consécutifs avec score suffisant
            self._confirmed = True                               # Confirme l'alerte LISA

        return gfm, ifm, slope, self._confirmed, score_signal, self._or_consecutive  # Retourne tous les indicateurs

    def reset(self):                                             # Réinitialise l'état interne du détecteur LISA
        self._gfm_current    = 0.0                               # Remet le GFM à zéro
        self._or_consecutive = 0                                 # Remet le compteur de confirmations à zéro
        self._confirmed      = False                             # Retire la confirmation d'alerte


# DÉTECTEUR KALMAN — adolescent


class KalmanDetector:
    """
    Détecteur Kalman gelé + fenêtre glissante.

    Correction importante :
    - Kalman se dégèle si le CGM descend pendant N_DEGEL steps consécutifs.
    - Après le dégel, il ne se regèle pas immédiatement dans le même step.
    - Tant que le CGM continue à descendre, Kalman reste en mode normal et suit le CGM.
    - Le gel est réactivé seulement si l'écart est positif ET que le CGM ne descend plus.
    """

    N_DEGEL = 5            # Nombre de steps consécutifs en descente CGM pour déclencher le dégel du filtre
    SLOPE_LOOKBACK_STEPS = 3  # Nombre de steps utilisés pour estimer la pente du CGM

    def __init__(self, first_cgm, sample_time):                  # Constructeur du filtre de Kalman
        self.dt = sample_time                                    # Pas de temps en minutes
        self.x  = np.array([[first_cgm], [0.0]])                 # État initial : [glycémie, dérivée] = [CGM0, 0]
        self.P  = np.eye(2) * 10                                 # Matrice de covariance initiale (incertitude élevée)
        self.F  = np.array([[1, self.dt], [0, 1]])               # Matrice de transition d'état (modèle cinématique)
        self.H  = np.array([[1, 0]])                             # Matrice d'observation (on mesure seulement la glycémie)
        self.Q  = Q_PROFILE                                      # Matrice de bruit de processus
        self.R  = R_PROFILE                                      # Matrice de bruit de mesure

        self.freeze = False                                      # Indicateur de gel du filtre (pas de correction)
        self._ecarts_fenetre = []                                 # Fenêtre glissante des écarts CGM - prédiction
        self._confirmed = False                                   # Indicateur d'alerte Kalman confirmée
        self._neg_slope_count = 0                                 # Compteur de steps consécutifs en descente CGM
        self._cgm_history = [float(first_cgm)]                   # Historique des mesures CGM
        self._prev_cgm = float(first_cgm)                        # Mesure CGM du step précédent

    def _slope_cgm(self):                                        # Calcule la pente du CGM sur les derniers steps
        if len(self._cgm_history) <= self.SLOPE_LOOKBACK_STEPS: # Pas assez d'historique pour calculer
            return 0.0                                           # Retourne une pente nulle
        return (self._cgm_history[-1] - self._cgm_history[-1 - self.SLOPE_LOOKBACK_STEPS]) / (
            self.SLOPE_LOOKBACK_STEPS * self.dt                  # Division par la durée en minutes pour avoir mg/dL/min
        )

    def update(self, cgm):                                       # Met à jour le filtre avec une nouvelle mesure CGM
        cgm = float(cgm)                                         # Convertit en float pour la sécurité
        self._cgm_history.append(cgm)                            # Enregistre la mesure dans l'historique

        x_pred = self.F @ self.x                                 # Prédiction de l'état : x_pred = F * x
        P_pred = self.F @ self.P @ self.F.T + self.Q             # Prédiction de la covariance : P_pred = F*P*F' + Q
        z = np.array([[cgm]])                                    # Vecteur de mesure (observation courante)

        p_libre = float((self.H @ x_pred)[0, 0])                 # Glycémie prédite par le filtre (libre de correction)
        ecart = float(cgm - p_libre)                             # Écart entre la mesure réelle et la prédiction
        residual = float((z - self.H @ x_pred)[0, 0])            # Résidu de la mesure (identique à ecart ici)

        # Variation instantanée du CGM.
        delta_cgm = cgm - self._prev_cgm                         # Variation de glycémie depuis le dernier step

        # Comptage des descentes CGM consécutives pour le dégel.
        if delta_cgm < 0:                                        # Si le CGM descend
            self._neg_slope_count += 1                           # Incrémente le compteur de descentes consécutives
        else:                                                    # Si le CGM monte ou stagne
            self._neg_slope_count = 0                            # Réinitialise le compteur

       
        just_degel = False                                       # Indicateur : le dégel vient juste de se produire
        if self.freeze and self._neg_slope_count >= self.N_DEGEL: # Condition de dégel : assez de steps en descente
            self.freeze = False                                  # Dégèle le filtre
            self._ecarts_fenetre = []                             # Vide la fenêtre des écarts
            self._confirmed = False                              # Retire la confirmation d'alerte
            self._neg_slope_count = 0                            # Réinitialise le compteur de descentes
            just_degel = True                                    # Marque que le dégel vient de se produire

       
        if (ecart > 0) and (not self.freeze) and (not just_degel) and (delta_cgm >= 0):  # Conditions de gel
            self.freeze = True                                   # Gèle le filtre (CGM au-dessus de la prédiction)
            self._neg_slope_count = 0                            # Réinitialise le compteur

        if self.freeze:                                          # Mode gelé : pas de correction Kalman
            self._ecarts_fenetre.append(max(0.0, ecart))         # Enregistre l'écart positif dans la fenêtre
            if len(self._ecarts_fenetre) > KALMAN_WINDOW:        # Si la fenêtre est trop grande
                self._ecarts_fenetre.pop(0)                      # Supprime le plus ancien écart
            score = float(np.mean(self._ecarts_fenetre)) if self._ecarts_fenetre else 0.0  # Score = moyenne des écarts

           
            self.x = x_pred                                      # Avance l'état sans correction (prédiction pure)
            self.P = P_pred                                      # Avance la covariance sans correction
        else:
            self._ecarts_fenetre = []                             # Vide la fenêtre en mode normal
            score = 0.0                                          # Score nul en mode normal

            # Mode normal : correction Kalman avec la mesure CGM.
            S = self.H @ P_pred @ self.H.T + self.R              # Innovation covariance S = H*P_pred*H' + R
            K = P_pred @ self.H.T @ np.linalg.inv(S)             # Gain de Kalman K = P_pred * H' * S^-1
            self.x = x_pred + K @ (z - self.H @ x_pred)          # Correction de l'état : x = x_pred + K * résidu
            self.P = (np.eye(2) - K @ self.H) @ P_pred           # Correction de la covariance : P = (I - K*H) * P_pred

        # Alerte directe : moyenne des résidus > seuil.

        if (not self.freeze) or just_degel or (delta_cgm < 0):   # Conditions pour retirer la confirmation
            self._confirmed = False                              # Pas d'alerte si non gelé, dégel récent, ou descente
        else:
            self._confirmed = bool(score > KALMAN_THRESHOLD)     # Alerte si score moyen dépasse le seuil

        self._prev_cgm = cgm                                     # Sauvegarde le CGM courant pour le prochain step
        return p_libre, residual, score, self._confirmed         # Retourne prédiction, résidu, score et alerte

    def set_freeze(self, freeze):                                 # Méthode de contrôle externe du gel (non utilisée)
        pass                                                     # Commande externe désactivée (fusion passive)

    def reset(self, cgm):                                        # Réinitialise le filtre Kalman avec une nouvelle valeur
        cgm = float(cgm)                                         # Convertit en float
        self.x  = np.array([[cgm], [0.0]])                       # Remet l'état initial à la valeur CGM donnée
        self.P  = np.eye(2) * 10                                 # Remet la covariance initiale à incertitude élevée
        self.freeze = False                                      # Déverrouille le filtre
        self._ecarts_fenetre = []                                 # Vide la fenêtre des écarts
        self._confirmed = False                                  # Retire la confirmation d'alerte
        self._neg_slope_count = 0                                # Remet le compteur de descentes à zéro
        self._cgm_history = [cgm]                                # Réinitialise l'historique CGM
        self._prev_cgm = cgm                                     # Met à jour la valeur CGM précédente



CAS_KK_LL = "K_alerte  + L_alerte  → PANNE CONFIRMÉE"           # Description du cas : les deux détecteurs alertent simultanément
CAS_KK_LS = "K_alerte  + L_silence → SUSPICION Kalman"          # Description du cas : seulement Kalman alerte
CAS_KS_LL = "K_silence + L_alerte  → SUSPICION LISA"            # Description du cas : seulement LISA alerte
CAS_KS_LS = "K_silence + L_silence → OK"                        # Description du cas : aucun détecteur n'alerte
CAS_SEQ   = "Alertes successives K→L ou L→K → SUSPICION SÉQUENTIELLE"  # Cas d'alertes décalées dans le temps


class FusionStateMachine:                                        # Machine à états pour la fusion des deux détecteurs

    def __init__(self):                                          # Constructeur de la machine à états de fusion
        self.etat      = "OK"                                    # État initial : pas d'anomalie détectée
        self.confiance = None                                    # Niveau de confiance courant (None = pas d'alerte)

        self.cmd_freeze = False                                  # Commande de gel envoyée à Kalman (fusion passive : inutilisée)
        self.cmd_reset  = False                                  # Commande de reset envoyée aux détecteurs (inutilisée)

        self.nb_panne         = 0                                # Compteur total de steps en état PANNE
        self.nb_suspicion     = 0                                # Compteur total de steps en état SUSPICION
        self.nb_suspicion_seq = 0                                # Compteur total de steps en SUSPICION SÉQUENTIELLE
        self.nb_ok            = 0                                # Compteur total de steps en état OK
        self.nb_panne_haute   = 0                                # Compteur de pannes à confiance HAUTE

        self.nb_cas = {                                          # Dictionnaire comptant les occurrences de chaque cas
            CAS_KK_LL: 0,                                        # Nombre de fois : K et L alertent ensemble
            CAS_KK_LS: 0,                                        # Nombre de fois : K alerte seul
            CAS_KS_LL: 0,                                        # Nombre de fois : L alerte seul
            CAS_KS_LS: 0,                                        # Nombre de fois : aucune alerte
            CAS_SEQ:   0,                                        # Nombre de fois : alertes séquentielles
        }

        self._k_first_alert_step = None                          # Step du premier déclenchement de l'alerte Kalman
        self._l_first_alert_step = None                          # Step du premier déclenchement de l'alerte LISA
        self.ecarts = []                                         # Historique des écarts temporels entre alertes K et L

        
        self._last_single_detector = None                        # Dernier détecteur ayant alerté seul ("K" ou "L")
        self._last_single_step     = None                        # Step où ce dernier détecteur a alerté seul
        self._last_single_window   = None                        # Fenêtre inter-bolus du dernier détecteur seul

    def _single_detector_courant(self, k_alert, l_alert):        # Identifie quel détecteur alerte seul
        if k_alert and not l_alert:                              # Seulement Kalman alerte
            return "K"                                           # Retourne "K"
        if l_alert and not k_alert:                              # Seulement LISA alerte
            return "L"                                           # Retourne "L"
        return None                                              # Aucun ou les deux alertent : retourne None

    def _est_sequentielle_interbolus(self, detector_now, step, bolus_window_start):  # Vérifie si une alerte est séquentielle
        if detector_now not in ("K", "L"):                       # Le détecteur courant doit être K ou L
            return False                                         # Sinon, pas séquentiel
        if self._last_single_detector not in ("K", "L"):         # Le dernier détecteur doit aussi être K ou L
            return False                                         # Sinon, pas séquentiel
        if detector_now == self._last_single_detector:           # Les deux alertes doivent venir de détecteurs différents
            return False                                         # Même détecteur : pas séquentiel
        if bolus_window_start is None or self._last_single_window is None:  # Les fenêtres doivent être définies
            return False                                         # Données manquantes : pas séquentiel
        if bolus_window_start != self._last_single_window:       # Les deux alertes doivent être dans la même fenêtre inter-bolus
            return False                                         # Fenêtres différentes : pas séquentiel
        return 0 < (step - self._last_single_step) <= SEQ_WINDOW_STEPS  # Vérifie que l'écart est dans la fenêtre SEQ

    def update(self, k_alert, l_alert, step, cgm_val, sample_time, bolus_window_start=None):  # Met à jour l'état de fusion
        
        self.cmd_freeze = False                                  # Réinitialise les commandes (fusion passive)
        self.cmd_reset  = False                                  # Idem pour reset

        detector_now = self._single_detector_courant(k_alert, l_alert)  # Identifie le détecteur actif ce step

        if k_alert and l_alert:                                  # CAS 1 : les deux détecteurs alertent → PANNE CONFIRMÉE
            cas = CAS_KK_LL                                      # Identifie le cas
            self.etat = "PANNE"                                  # Passe en état PANNE
            self.confiance = "HAUTE"                             # Confiance maximale

            if self._k_first_alert_step is None:                 # Enregistre le premier step d'alerte Kalman
                self._k_first_alert_step = step                  # (si pas encore enregistré)
            if self._l_first_alert_step is None:                 # Enregistre le premier step d'alerte LISA
                self._l_first_alert_step = step                  # (si pas encore enregistré)

            self._enregistrer_ecart(step, "HAUTE")               # Enregistre l'écart entre les deux alertes

            
            self._last_single_detector = None                    # Réinitialise le suivi du détecteur seul
            self._last_single_step     = None                    # Réinitialise le step du détecteur seul
            self._last_single_window   = None                    # Réinitialise la fenêtre du détecteur seul

        elif detector_now and self._est_sequentielle_interbolus(detector_now, step, bolus_window_start):  # CAS 2 : alertes séquentielles
            cas = CAS_SEQ                                        # Identifie le cas séquentiel
            self.etat = "SUSPICION_SEQUENTIELLE"                 # Passe en état de suspicion séquentielle
            self.confiance = "SEQUENTIELLE"                      # Niveau de confiance intermédiaire

            if detector_now == "K" and self._k_first_alert_step is None:  # Si c'est Kalman qui alerte maintenant
                self._k_first_alert_step = step                  # Enregistre le premier step d'alerte Kalman
            if detector_now == "L" and self._l_first_alert_step is None:  # Si c'est LISA qui alerte maintenant
                self._l_first_alert_step = step                  # Enregistre le premier step d'alerte LISA

            self._enregistrer_ecart(step, "SEQUENTIELLE")        # Enregistre l'écart temporel entre les alertes

           
            self._last_single_detector = detector_now            # Met à jour le dernier détecteur seul
            self._last_single_step     = step                    # Met à jour le step du dernier détecteur seul
            self._last_single_window   = bolus_window_start      # Met à jour la fenêtre inter-bolus

        elif k_alert and not l_alert:                            # CAS 3 : seulement Kalman alerte → SUSPICION Kalman
            cas = CAS_KK_LS                                      # Identifie le cas
            self.etat = "SUSPICION"                              # Passe en état de suspicion
            self.confiance = None                                 # Pas de niveau de confiance défini

            if self._k_first_alert_step is None:                 # Enregistre le premier step d'alerte Kalman
                self._k_first_alert_step = step                  # (si pas encore enregistré)

            self._last_single_detector = "K"                     # Met à jour : Kalman est le dernier détecteur seul
            self._last_single_step     = step                    # Enregistre le step courant
            self._last_single_window   = bolus_window_start      # Enregistre la fenêtre inter-bolus

        elif l_alert and not k_alert:                            # CAS 4 : seulement LISA alerte → SUSPICION LISA
            cas = CAS_KS_LL                                      # Identifie le cas
            self.etat = "SUSPICION"                              # Passe en état de suspicion
            self.confiance = None                                 # Pas de niveau de confiance défini

            if self._l_first_alert_step is None:                 # Enregistre le premier step d'alerte LISA
                self._l_first_alert_step = step                  # (si pas encore enregistré)

            self._last_single_detector = "L"                     # Met à jour : LISA est le dernier détecteur seul
            self._last_single_step     = step                    # Enregistre le step courant
            self._last_single_window   = bolus_window_start      # Enregistre la fenêtre inter-bolus

        else:                                                    # CAS 5 : aucune alerte → état OK
            cas = CAS_KS_LS                                      # Identifie le cas
            self.etat = "OK"                                     # Passe en état OK
            self.confiance = None                                 # Pas de confiance (pas d'alerte)
            self._reset_first_alerts()                           # Réinitialise les premiers steps d'alerte

        self.nb_cas[cas] += 1                                    # Incrémente le compteur du cas détecté

        if self.etat == "PANNE":                                 # Mise à jour des statistiques selon l'état
            self.nb_panne += 1                                   # Incrémente le compteur de pannes
            self.nb_panne_haute += 1                             # Incrémente le compteur de pannes à haute confiance
        elif self.etat == "SUSPICION_SEQUENTIELLE":              # État suspicion séquentielle
            self.nb_suspicion_seq += 1                           # Incrémente le compteur correspondant
        elif self.etat == "SUSPICION":                           # État suspicion simple
            self.nb_suspicion += 1                               # Incrémente le compteur de suspicions
        else:                                                    # État OK
            self.nb_ok += 1                                      # Incrémente le compteur d'états OK

        return self.etat, self.confiance, cas                    # Retourne l'état, le niveau de confiance et le cas

    def _enregistrer_ecart(self, step, confiance):               # Enregistre l'écart temporel entre les deux alertes
        if (self._k_first_alert_step is not None and             # Si les deux alertes ont un step enregistré
                self._l_first_alert_step is not None):
            ecart = self._l_first_alert_step - self._k_first_alert_step  # Calcule l'écart en steps (positif = L après K)
            self.ecarts.append((step, ecart, confiance))         # Ajoute (step, écart, confiance) à l'historique
        elif self._k_first_alert_step is not None:               # Seulement Kalman a alerté
            self.ecarts.append((step, None, confiance))          # Enregistre avec écart None (LISA pas encore alerté)
        elif self._l_first_alert_step is not None:               # Seulement LISA a alerté
            self.ecarts.append((step, None, confiance))          # Enregistre avec écart None (Kalman pas encore alerté)

    def _reset_first_alerts(self):                               # Réinitialise les premiers steps d'alerte des deux détecteurs
        self._k_first_alert_step = None                          # Oublie le premier step d'alerte Kalman
        self._l_first_alert_step = None                          # Oublie le premier step d'alerte LISA


# ============================================================
# INITIALISATION PATIENT & ENVIRONNEMENT
# ============================================================

patient    = T1DPatient.withName(PATIENT_NAME)                   # Charge le profil physiologique de l'adolescent
sensor     = CGMSensor.withName("Dexcom", seed=1)                # Initialise le capteur CGM Dexcom avec graine aléatoire 1
pump       = InsulinPump.withName("Insulet")                     # Initialise le modèle de pompe Insulet
start_time = datetime.now()                                      # Enregistre l'heure de début de simulation

env_tmp     = T1DSimEnv(patient=patient, sensor=sensor, pump=pump,  # Crée un environnement temporaire pour lire le sample_time
                        scenario=AdoScenario(start_time, [], 3))
sample_time = env_tmp.sample_time                                # Récupère l'intervalle de temps de l'environnement (en minutes)
del env_tmp                                                      # Supprime l'environnement temporaire (plus nécessaire)

REPAS    = build_repas_for_horizon(sample_time, horizon_steps=10000)  # Génère le planning de repas complet pour 10000 steps
patient  = T1DPatient.withName(PATIENT_NAME)                     # Recrée le patient (réinitialisation propre)
sensor   = CGMSensor.withName("Dexcom", seed=1)                  # Recrée le capteur CGM (même graine pour reproductibilité)
pump     = InsulinPump.withName("Insulet")                       # Recrée la pompe à insuline
scenario = AdoScenario(start_time, REPAS, sample_time)           # Crée le scénario complet avec les repas planifiés
env      = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)  # Crée l'environnement de simulation final

for r in REPAS:                                                  # Ajoute le numéro de step à chaque repas
    r["step"] = int(r["min"] / sample_time)                      # Convertit les minutes en steps

BOLUS_CARBS_STEPS = {max(0, r["step"] - PREBOLUS_STEPS): r["carbs"] for r in REPAS}  # Dictionnaire {step_bolus: glucides} avec pré-bolus
BOLUS_STEPS       = BOLUS_CARBS_STEPS                            # Alias pour les steps où un bolus est prévu


def get_inter_bolus_window_start(step, bolus_steps_dict):        # Retourne le step du bolus précédent (fenêtre inter-bolus)
    bs = sorted(bolus_steps_dict.keys())                         # Trie les steps de bolus en ordre croissant
    if not bs or step < bs[0]:                                   # Si aucun bolus ou avant le premier bolus
        return None                                              # Pas de fenêtre définie
    lo, hi, idx = 0, len(bs) - 1, 0                             # Initialise la recherche binaire
    while lo <= hi:                                              # Recherche binaire du dernier bolus ≤ step courant
        mid = (lo + hi) // 2                                     # Calcule l'indice du milieu
        if bs[mid] <= step: idx = mid; lo = mid + 1             # Si bs[mid] ≤ step, continue à droite
        else:               hi  = mid - 1                       # Sinon, continue à gauche
    return bs[idx]                                               # Retourne le step du bolus le plus récent ≤ step


def update_bolus_steps(current_step, scenario, sample_time):     # Met à jour le dictionnaire BOLUS_CARBS_STEPS si nécessaire
    scenario.extend_if_needed(current_step, sample_time, horizon=500)  # Étend le scénario de repas si l'horizon approche
    for r in scenario.repas:                                     # Parcourt tous les repas du scénario
        s_r = r.get("step")                                      # Récupère le step du repas
        if s_r is None: continue                                 # Ignore les repas sans step défini
        bolus_step = max(0, s_r - PREBOLUS_STEPS)                # Calcule le step du pré-bolus correspondant
        if bolus_step not in BOLUS_CARBS_STEPS:                  # Si ce step n'est pas encore dans le dictionnaire
            BOLUS_CARBS_STEPS[bolus_step] = r["carbs"]           # Ajoute le pré-bolus au dictionnaire


MAX_SIM_STEPS = None  # Nombre maximum de steps (None = simulation sans fin jusqu'à Ctrl+C)

print(f"\n{'='*72}")                                             # Affiche une ligne de séparation
print(f"  FUSION KALMAN + LISA v1 — Détection de panne pompe ({PATIENT_NAME})")  # Titre du programme
print(f"{'='*72}")                                               # Affiche une ligne de séparation
print(f"  Sample time      : {sample_time} min")                 # Affiche l'intervalle de temps entre steps
print(f"  LISA LW / SW     : {LISA_LW} steps ({LISA_LW*sample_time:.0f} min) / "  # Affiche les tailles des fenêtres LISA
      f"{LISA_SW} steps ({LISA_SW*sample_time:.0f} min)")
print(f"  LISA GFM seuil   : {LISA_GFM_THRESHOLD}")             # Affiche le seuil GFM de LISA
print(f"  LISA IFM seuil   : {LISA_IFM_THRESHOLD}")             # Affiche le seuil IFM de LISA
print(f"  LISA Pente seuil : {LISA_SLOPE_THRESHOLD} mg/dL/min") # Affiche le seuil de pente de LISA
print(f"  LISA Score min   : {LISA_SCORE_MIN} / 4 pts  (GFM×2 + IFM×1 + Pente×1)")  # Affiche le score minimum LISA
print(f"  LISA Confirmation: {LISA_CONFIRMATION_STEPS} steps = {LISA_CONFIRMATION_STEPS*sample_time:.0f} min")  # Affiche la confirmation LISA
print(f"  Kalman seuil     : {KALMAN_THRESHOLD}  |  fenêtre : {KALMAN_WINDOW} steps ({KALMAN_WINDOW*sample_time:.0f} min)")  # Affiche les paramètres Kalman
print(f"  Kalman décision  : directe si moyenne résidus > seuil | dégel après {KalmanDetector.N_DEGEL} steps descendants")  # Règle de décision Kalman
print(f"  Cycle pannes     : {PANNE_DUREE} steps HS ({PANNE_DUREE*sample_time:.0f} min) → "  # Affiche le cycle de pannes simulées
      f"{MARCHE_DUREE} steps OK ({MARCHE_DUREE*sample_time:.0f} min)")
print(f"  Repas            : {len(REPAS_JOURNEE)} repas/jour, cycle {MINUTES_PAR_JOUR} min")  # Affiche le nombre de repas
print(f"  Bolus            : ICR={ICR}g/U | ISF={ISF}mg/dL/U")  # Affiche les paramètres de calcul du bolus
print("  Durée évaluée    : simulation continue (arrêt avec Ctrl+C)")  # Indique la durée de la simulation
print("  Arrêt            : Ctrl+C")                             # Rappel de la commande d'arrêt
print(f"{'='*72}\n")                                             # Ligne de séparation finale

print(f"{'Step':>5} {'Heure':>6} {'CGM':>7} "                   # Affiche l'en-tête de la table console (colonnes 1-3)
      f"{'K_alr':>7} {'L_alr':>7} {'Score':>8} {'Consec':>7} "  # Colonnes alertes Kalman/LISA, score et consécutifs
      f"{'GFM':>10} {'IFM':>7} {'Pente':>7} "                   # Colonnes indicateurs LISA
      f"{'Fusion':>22} {'Confiance':>14} {'Détecteur':>15} "     # Colonnes état fusion, confiance, détecteur
      f"{'Cas':>42} {'Réel':>8}")                                # Colonnes cas de fusion et état réel de la pompe
print("─" * 170)                                                 # Ligne de séparation sous l'en-tête


# ============================================================
# GRAPHE — 4 panneaux
# ============================================================

plt.style.use("default")                                         # Utilise le style matplotlib par défaut (fond blanc)

C_BG        = "white"          # Couleur de fond des panneaux graphiques
C_GRID      = "#cccccc"        # Couleur de la grille (gris clair)
C_SPINE     = "#aaaaaa"        # Couleur des bordures des axes
C_TICK      = "#333333"        # Couleur des graduations et labels des axes
C_CGM       = "#005f5f"        # Couleur de la courbe CGM (vert-bleu, distinct de l'enfant en bleu)
C_PRED      = "#b35900"        # Couleur de la prédiction Kalman (orange foncé)
C_KSCORE    = "#8b0000"        # Couleur du score Kalman (rouge foncé)
C_GFM       = "#4b0082"        # Couleur de la courbe GFM (violet indigo)
C_SLOPE     = "#005f5f"        # Couleur de la courbe de pente (vert-bleu)
C_IFM       = "#7a3500"        # Couleur de la courbe IFM (brun)
C_OR_BRUT   = "#aaaaaa"        # Couleur des steps consécutifs (gris)
C_SEUIL_K   = "#cc0000"        # Couleur du seuil Kalman (rouge)
C_SEUIL_GFM = "#4b0082"        # Couleur du seuil GFM (violet)
C_SEUIL_SLP = "#005f5f"        # Couleur du seuil de pente (vert-bleu)
C_SEUIL_IFM = "#7a3500"        # Couleur du seuil IFM (brun)
C_MEAL      = "#cc5500"        # Couleur des marqueurs de repas (orange)
C_BOLUS_OK  = "#003f8a"        # Couleur des bolus injectés (bleu foncé)
C_BOLUS_FAIL= "#cc0000"        # Couleur des bolus manqués (rouge)
C_HAUTE     = "#cc0000"        # Couleur des alertes de panne confirmée (rouge)
C_SUSPICION = "#005f5f"        # Couleur des alertes de suspicion simple (vert-bleu)
C_SEQ       = "#b87700"        # Couleur des alertes de suspicion séquentielle (or foncé)
C_BAND_FAIL = "#ffcccc"        # Couleur des bandes de fond pendant une panne réelle (rose clair)
C_BAND_OK   = "#e6f7ee"        # Couleur des bandes de fond pendant le fonctionnement normal (vert très clair)
C_ZONE_CIBLE= "#e8f4fd"        # Couleur de la zone cible glycémique [70-180 mg/dL] (bleu très clair)

fig = plt.figure(figsize=(16, 10))     # Crée une figure matplotlib de 16x10 pouces
fig.patch.set_facecolor(C_BG)          # Définit la couleur de fond de la figure entière

gs = gridspec.GridSpec(4, 1, figure=fig,           # Crée une grille de 4 lignes et 1 colonne pour les sous-graphes
                       height_ratios=[3, 1, 1, 1], hspace=0.08)  # Ratios de hauteur et espacement vertical

ax_cgm = fig.add_subplot(gs[0])                    # Panneau 1 (le plus grand) : courbe de glycémie CGM
ax_k   = fig.add_subplot(gs[1], sharex=ax_cgm)    # Panneau 2 : score Kalman (axe X partagé avec CGM)
ax_l   = fig.add_subplot(gs[2], sharex=ax_cgm)    # Panneau 3 : indicateurs LISA (axe X partagé)
ax_fus = fig.add_subplot(gs[3], sharex=ax_cgm)    # Panneau 4 : décisions de fusion (axe X partagé)
AXES   = [ax_cgm, ax_k, ax_l, ax_fus]             # Liste des 4 axes pour les boucles de mise en forme

for ax in AXES:                                    # Applique la mise en forme commune à tous les axes
    ax.set_facecolor(C_BG)                         # Fond blanc pour chaque panneau
    ax.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)  # Grille grise semi-transparente et fine
    ax.tick_params(colors=C_TICK, labelsize=7)     # Graduations en gris foncé, taille 7
    for s in ["top", "right"]:   ax.spines[s].set_visible(False)  # Cache les bordures haut et droite
    for s in ["bottom", "left"]: ax.spines[s].set_color(C_SPINE)  # Colore les bordures bas et gauche

# Panneau CGM
line_cgm,  = ax_cgm.plot([], [], color=C_CGM,  lw=2.5, zorder=3)  # Ligne CGM : initialisée vide, sera mise à jour
line_pred, = ax_cgm.plot([], [], color=C_PRED, ls="--", lw=1.5, zorder=3, alpha=0.85)  # Ligne prédiction Kalman (tirets)
sc_meals       = ax_cgm.scatter([], [], marker="^", s=160, color=C_MEAL,       zorder=6)  # Triangles pour les repas
sc_bolus_ok    = ax_cgm.scatter([], [], marker="o", s=120, color=C_BOLUS_OK,   zorder=6)  # Cercles pour bolus injectés
sc_bolus_fail  = ax_cgm.scatter([], [], marker="x", s=160, color=C_BOLUS_FAIL, zorder=6, lw=2.5)  # Croix pour bolus manqués
sc_panne_haute = ax_cgm.scatter([], [], marker="D", s=180, color=C_HAUTE,      zorder=9, lw=1.5)  # Diamants pour panne confirmée
sc_suspicion   = ax_cgm.scatter([], [], marker="o", s=80,  color=C_SUSPICION,  zorder=8,  # Cercles vides pour suspicion simple
                                facecolors="none", lw=2.0)
sc_susp_seq    = ax_cgm.scatter([], [], marker="s", s=100, color=C_SEQ,        zorder=8,  # Carrés vides pour suspicion séquentielle
                                facecolors="none", lw=2.0)

ax_cgm.axhline(70,  color=C_BOLUS_OK,   ls="--", lw=1.2, alpha=0.7)   # Ligne horizontale à 70 mg/dL (seuil hypoglycémie)
ax_cgm.axhline(180, color=C_BOLUS_FAIL, ls="--", lw=1.2, alpha=0.7)   # Ligne horizontale à 180 mg/dL (seuil hyperglycémie)
ax_cgm.axhspan(70, 180, alpha=0.08, color=C_ZONE_CIBLE)                # Zone colorée entre 70 et 180 mg/dL (zone cible)
ax_cgm.set_ylim(40, 450)                                                # Fixe l'axe Y entre 40 et 450 mg/dL
ax_cgm.set_ylabel("Glycémie (mg/dL)", color=C_TICK, fontsize=8)        # Label de l'axe Y du panneau CGM

legend_elems = [                                                 # Éléments pour la légende du panneau CGM
    Line2D([0],[0], color=C_CGM,   lw=2.5,          label="CGM"),       # Légende : ligne CGM
    Line2D([0],[0], color=C_PRED,  lw=1.5, ls="--",  label="Prédiction Kalman"),  # Légende : prédiction Kalman
    Patch(facecolor=C_BAND_FAIL, alpha=0.5,           label="Pompe HS (réel)"),    # Légende : bande panne réelle
    Patch(facecolor=C_BAND_OK,   alpha=0.5,           label="Pompe OK (réel)"),    # Légende : bande fonctionnement OK
    Line2D([0],[0], marker="^", color="w", markerfacecolor=C_MEAL,     ms=8, ls="None", label="Repas"),  # Légende : repas
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BOLUS_OK, ms=7, ls="None", label="Bolus injecté"),  # Légende : bolus injecté
    Line2D([0],[0], marker="x", color=C_BOLUS_FAIL, ms=8, ls="None",  label="Bolus manqué"),  # Légende : bolus manqué
    Line2D([0],[0], marker="D", color=C_HAUTE,      ms=8, ls="None",  label="PANNE CONFIRMÉE (K+L)"),  # Légende : panne confirmée
    Line2D([0],[0], marker="o", color=C_SUSPICION,  ms=8, ls="None",                           # Légende : suspicion simple
           markerfacecolor="none", label="SUSPICION (1 détecteur)"),
    Line2D([0],[0], marker="s", color=C_SEQ,        ms=8, ls="None",                           # Légende : suspicion séquentielle
           markerfacecolor="none", label="SUSPICION SÉQUENTIELLE"),
]
ax_cgm.legend(handles=legend_elems, loc="upper left",           # Affiche la légende en haut à gauche
              fontsize=7, facecolor="white", edgecolor=C_SPINE, ncol=3)  # 3 colonnes, fond blanc, petite police

# Panneau Kalman
line_kscore, = ax_k.plot([], [], color=C_KSCORE, lw=2.0)         # Ligne du score Kalman (initialisée vide)
ax_k.axhline(KALMAN_THRESHOLD, color=C_SEUIL_K, ls="--", lw=1.2, alpha=0.85)  # Ligne de seuil Kalman (tirets rouges)
ax_k.set_ylim(0, max(80, KALMAN_THRESHOLD * 1.8))                # Axe Y avec marge au-dessus du seuil
ax_k.set_ylabel("Score\nKalman", color=C_TICK, fontsize=7)       # Label de l'axe Y du panneau Kalman

# Panneau LISA
line_gfm,    = ax_l.plot([], [], color=C_GFM,    lw=2.0,       label="GFM/100")  # Courbe GFM normalisée
line_ifm,    = ax_l.plot([], [], color=C_IFM,    lw=1.5, ls="-.", label="IFM")   # Courbe IFM (tirets-points)
line_slope,  = ax_l.plot([], [], color=C_SLOPE,   lw=1.5,       label="Pente")   # Courbe de pente du CGM
line_consec, = ax_l.plot([], [], color=C_OR_BRUT, lw=1.0, ls=":",  # Courbe des steps consécutifs normalisée (pointillés)
                          label=f"Consécutifs/{LISA_CONFIRMATION_STEPS}")

ax_l.axhline(LISA_GFM_THRESHOLD / 100, color=C_SEUIL_GFM, ls="--", lw=1.2, alpha=0.85)  # Seuil GFM normalisé (tirets violets)
ax_l.axhline(LISA_IFM_THRESHOLD,        color=C_SEUIL_IFM, ls="--", lw=1.2, alpha=0.85)  # Seuil IFM (tirets bruns)
ax_l.axhline(LISA_SLOPE_THRESHOLD,      color=C_SEUIL_SLP, ls="--", lw=1.2, alpha=0.85)  # Seuil de pente (tirets vert-bleu)
ax_l.axhline(1.0, color=C_OR_BRUT, ls=":", lw=1.0, alpha=0.6, label="Seuil confirmation")  # Seuil de confirmation (pointillés gris)
ax_l.set_ylim(-2, 12)                                            # Axe Y du panneau LISA entre -2 et 12
ax_l.set_ylabel("Indicateurs\nLISA", color=C_TICK, fontsize=7)  # Label de l'axe Y du panneau LISA
ax_l.legend(loc="upper left", fontsize=6, facecolor="white", edgecolor=C_SPINE, ncol=2)  # Légende LISA à 2 colonnes

# Panneau fusion
ax_fus.set_ylim(-0.5, 3.5)                                       # Axe Y du panneau fusion : de -0.5 à 3.5
ax_fus.set_yticks([0, 1, 1.5, 2, 3])                             # Graduations correspondant aux états de fusion
ax_fus.set_yticklabels(["OK", "SUSP.", "SUSP.\nSÉQ.", "PANNE\nfusion", "PANNE\nréelle"],  # Labels des états
                        color=C_TICK, fontsize=6)
ax_fus.set_ylabel("Décision", color=C_TICK, fontsize=7)          # Label de l'axe Y du panneau fusion
ax_fus.set_xlabel("Heure simulée", color=C_TICK, fontsize=8)     # Label de l'axe X commun

fusion_level_history = []   # Historique du niveau de fusion numérique (0=OK, 1=Susp, 1.5=SeqSusp, 2=Panne)
real_level_history   = []   # Historique du niveau réel de la pompe (3=panne réelle, 2.85=OK)

txt_etat  = ax_cgm.text(0.99, 0.97, "", transform=ax_cgm.transAxes,  # Texte principal : glycémie + état fusion (haut droite)
                         color=C_CGM, fontsize=13, ha="right", va="top",
                         fontfamily="monospace", fontweight="bold")
txt_info  = ax_cgm.text(0.99, 0.89, "", transform=ax_cgm.transAxes,  # Texte info : step, alertes, indicateurs (milieu droite)
                         color="#444444", fontsize=8, ha="right", va="top")
txt_cas   = ax_cgm.text(0.99, 0.82, "", transform=ax_cgm.transAxes,  # Texte cas : cas fusion, confiance, écart (bas droite)
                         color="#7a5c00", fontsize=8, ha="right", va="top",
                         fontfamily="monospace")
txt_alarm = ax_cgm.text(0.5, 0.50, "", transform=ax_cgm.transAxes,   # Texte alarme : message centré dans le panneau CGM
                         color=C_HAUTE, fontsize=12, ha="center", va="center",
                         fontfamily="monospace", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe5e5",  # Boîte rouge pour l'alarme
                                   edgecolor=C_HAUTE, alpha=0.0))

fig.suptitle(                                                    # Titre principal de la figure (au-dessus de tous les panneaux)
    f"Fusion Kalman + LISA v1 — Détection de panne pompe ({PATIENT_NAME})\n"  # Titre : nom du système et patient
    f"LISA: score≥{LISA_SCORE_MIN} + confirmation {LISA_CONFIRMATION_STEPS} steps "  # Paramètres LISA résumés
    f"({LISA_CONFIRMATION_STEPS*3}min) | Kalman: fenêtre {KALMAN_WINDOW} steps, "    # Paramètres Kalman résumés
    f"seuil {KALMAN_THRESHOLD}  —  Ctrl+C",                     # Rappel de l'arrêt
    fontsize=10, color="#111111", fontfamily="monospace"          # Style du titre
)

failure_bands = {ax: [] for ax in AXES}  # Dictionnaire stockant les bandes colorées de panne pour chaque axe
fill_ref      = [None]                   # Référence à l'objet fill_between du CGM (pour pouvoir le supprimer/recréer)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # Ajuste les marges en laissant de la place pour le titre en haut
plt.ion()                                # Active le mode interactif de matplotlib (mise à jour en temps réel)
plt.show()                               # Affiche la fenêtre graphique


# ============================================================
# INITIALISATION BOUCLE
# ============================================================

s    = env.reset()       # Réinitialise l'environnement de simulation et retourne l'observation initiale
obs  = s.observation     # Extrait l'observation (contient la valeur CGM, etc.)
done = s.done            # Indicateur de fin d'épisode

index      = 0           # Compteur de steps (commence à 0)
prev_state = None        # État précédent de la pompe (pour détecter les transitions panne/marche)

kalman = KalmanDetector(float(obs.CGM), sample_time)  # Initialise le détecteur Kalman avec la glycémie initiale
lisa   = LISADetector(sample_time)                    # Initialise le détecteur LISA
fusion = FusionStateMachine()                         # Initialise la machine à états de fusion

times             = []   # Historique des labels de temps formatés ("HH:MM")
cgm_vals          = []   # Historique des valeurs CGM mesurées
pred_vals         = []   # Historique des prédictions Kalman
pump_states       = []   # Historique de l'état réel de la pompe (True=panne, False=OK)
k_scores          = []   # Historique des scores Kalman
k_alerts_h        = []   # Historique des alertes Kalman (True/False)
l_gfm_h           = []   # Historique des valeurs GFM de LISA
l_ifm_h           = []   # Historique des valeurs IFM de LISA
l_slope_h         = []   # Historique des pentes CGM calculées par LISA
l_consec_h        = []   # Historique du compteur de steps consécutifs LISA
l_or_brut_h       = []   # Historique des signaux OR bruts (score ≥ seuil) de LISA
l_alerts_h        = []   # Historique des alertes LISA confirmées (True/False)
fusion_states     = []   # Historique des états de fusion ("OK", "SUSPICION", etc.)
cas_history       = []   # Historique des cas de fusion identifiés
confiance_history = []   # Historique des niveaux de confiance de la fusion

MAX_PTS = 80             # Nombre maximum de points affichés simultanément sur les graphiques (fenêtre glissante)

bolus_total_prevu   = 0   # Compteur cumulé de bolus prévus (repas attendus)
bolus_total_injecte = 0   # Compteur cumulé de bolus effectivement injectés
bolus_total_manque  = 0   # Compteur cumulé de bolus manqués (pompe en panne au moment du bolus)
insuline_injectee_U = 0.0 # Total d'insuline injectée en unités (basale + bolus réels)
insuline_manquee_U  = 0.0 # Total d'insuline manquée en unités (bolus non délivrés)
insuline_basale_U   = 0.0 # Total d'insuline basale injectée en unités


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

STOP_REASON = "Ctrl+C"  # Raison d'arrêt par défaut : interruption clavier

try:
    while True:                                                  # Boucle infinie (arrêt avec Ctrl+C)
        update_bolus_steps(index, scenario, sample_time)         # Étend le planning de repas si nécessaire
        cgm_val    = float(obs.CGM)                              # Récupère la valeur CGM de l'observation courante
        sim_min    = index * sample_time                         # Calcule le temps simulé écoulé en minutes
        time_label = (start_time + timedelta(minutes=sim_min)).strftime("%H:%M")  # Formate en "HH:MM"

        carbs_for_bolus = BOLUS_CARBS_STEPS.get(index, 0)        # Récupère les glucides prévus à ce step (0 si aucun repas)
        bolus_plan      = calculer_bolus(carbs_for_bolus, cgm_val) if carbs_for_bolus > 0 else 0.0  # Calcule le bolus prévu
        failed          = pump_is_failed(index)                  # Détermine si la pompe est en panne à ce step

        if prev_state is not None and failed != prev_state:      # Détecte les transitions d'état de la pompe
            sep = "─" * 110                                      # Ligne de séparation visuelle dans la console
            if failed:                                           # Si la pompe vient de tomber en panne
                print(f"\n{sep}\n  [!] PANNE POMPE RÉELLE — step {index} ({time_label})\n{sep}\n")  # Message de panne
            else:                                                # Si la pompe vient d'être rétablie
                print(f"\n{sep}\n  [v] POMPE RÉTABLIE — step {index} ({time_label})\n{sep}\n")     # Message de rétablissement
        prev_state = failed                                      # Mémorise l'état courant pour le prochain step

        # ── Basal dynamique ado ──────────────────────────────────────────────
        dynamic_basal = BASAL                                    # Commence avec le débit basal nominal
        if cgm_val > 170: dynamic_basal += 0.007                 # Légère augmentation si glycémie > 170 mg/dL
        if cgm_val > 210: dynamic_basal += 0.012                 # Augmentation supplémentaire si > 210 mg/dL (hyperglycémie)
        if cgm_val < 100: dynamic_basal -= 0.007                 # Légère réduction si glycémie < 100 mg/dL
        if cgm_val <  85: dynamic_basal -= 0.012                 # Réduction supplémentaire si < 85 mg/dL (risque hypoglycémie)
        dynamic_basal = max(0.01, dynamic_basal)                 # Garde un débit basal minimum de 0.01 U/min

        if bolus_plan == 0.0 and cgm_val > 200.0:                # Si pas de repas prévu mais hyperglycémie > 200 mg/dL
            bolus_final = min(3.0, max(0.0, (cgm_val - CGM_CIBLE) / ISF))  # Bolus de correction plafonné à 3 U
        else:                                                    # Sinon, utilise le bolus repas calculé
            bolus_final = bolus_plan                             # Bolus final = bolus repas planifié

        action = (Action(basal=0.0, bolus=0.0) if failed else    # Si panne : aucune action (pompe arrêtée)
                  Action(basal=dynamic_basal, bolus=bolus_final)) # Si OK : envoie basal dynamique + bolus final

        insulin_dose_step = (0.0 if failed else                  # Dose totale d'insuline pour ce step (0 si panne)
                             dynamic_basal * sample_time + bolus_final)  # Basal (en U) + bolus (en U)

        # ── Kalman ──────────────────────────────────────────────────────────
        pred, residual, k_score, k_alert = kalman.update(cgm_val)  # Met à jour Kalman et récupère prédiction, résidu, score, alerte

        # ── LISA ────────────────────────────────────────────────────────────
        gfm, ifm, slope, l_alert, or_brut, consec = lisa.update(cgm_val, insulin_dose_step)  # Met à jour LISA et récupère ses indicateurs

        # ── Fusion ──────────────────────────────────────────────────────────
        bolus_window_start = get_inter_bolus_window_start(index, BOLUS_CARBS_STEPS)  # Détermine la fenêtre inter-bolus courante
        etat_fusion, confiance, cas = fusion.update(             # Met à jour la machine à états de fusion
            k_alert, l_alert, index, cgm_val, sample_time, bolus_window_start  # Paramètres : alertes, step, CGM, temps, fenêtre
        )

        # ── Détecteur responsable ────────────────────────────────────────────
        if k_alert and l_alert:    detecteur_txt = "Kalman + LISA"  # Les deux détecteurs alertent
        elif k_alert:              detecteur_txt = "Kalman seul"    # Seulement Kalman alerte
        elif l_alert:              detecteur_txt = "LISA seule"     # Seulement LISA alerte
        else:                      detecteur_txt = "Aucun"          # Aucun détecteur n'alerte

        # ── Notifications ────────────────────────────────────────────────────
        if etat_fusion == "PANNE":                               # Si panne confirmée
            notifier_panne(index)                                # Envoie la notification de panne urgente
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":            # Si suspicion séquentielle
            _ecart_seq = 0                                       # Initialise l'écart à 0
            if (fusion._k_first_alert_step is not None and       # Si les deux steps d'alerte sont définis
                    fusion._l_first_alert_step is not None):
                _ecart_seq = abs(fusion._l_first_alert_step - fusion._k_first_alert_step)  # Calcule l'écart en steps
            notifier_suspicion_sequentielle(index, _ecart_seq, sample_time)  # Envoie la notification séquentielle
        elif etat_fusion == "SUSPICION":                         # Si suspicion simple
            notifier_suspicion(index)                            # Envoie la notification de suspicion

        # ── Commandes Kalman depuis la fusion ────────────────────────────────
        # Fusion passive : on ne commande ni Kalman ni LISA.
        # Les deux détecteurs restent indépendants.
        # Ancienne logique supprimée :
        #   - pas de kalman.reset(cgm_val) demandé par la fusion
        #   - pas de lisa.reset() demandé par la fusion
        #   - pas de kalman.set_freeze(...) demandé par la fusion

        # ── Suivi bolus prévu/manqué (sans calcul des métriques VP/FP/FN) ─────
        bolus_attendu_ce_step = index in BOLUS_CARBS_STEPS        # True si un bolus était prévu à ce step
        bolus_manque_ce_step  = failed and bolus_attendu_ce_step  # True si la pompe est en panne et un bolus était prévu

        # ── Console ──────────────────────────────────────────────────────────
        conf_txt = confiance if confiance else "—"               # Texte du niveau de confiance ("—" si None)
        or_txt   = "OR" if or_brut else "  "                     # Texte pour le signal OR brut LISA ("OR" ou vide)
        print(                                                   # Affiche une ligne de données dans la console
            f"[{index:>4}] {time_label} "                        # Numéro de step et heure simulée
            f"{cgm_val:>7.1f} "                                  # Valeur CGM
            f"{'OUI' if k_alert else 'NON':>7} "                 # Alerte Kalman (OUI/NON)
            f"{'OUI' if l_alert else 'NON':>7} "                 # Alerte LISA (OUI/NON)
            f"{or_txt:>8} {consec:>7} "                          # Signal OR brut et compteur consécutif
            f"{gfm:>10.1f} "                                     # Valeur GFM
            f"{ifm:>7.3f} "                                      # Valeur IFM
            f"{slope:>7.2f} "                                    # Pente CGM
            f"{etat_fusion:>22} "                                # État de fusion
            f"{conf_txt:>14} "                                   # Niveau de confiance
            f"{detecteur_txt:>15} "                              # Nom du détecteur responsable
            f"{'[B=' + str(bolus_plan) + 'U]' if bolus_plan > 0 else '':>8} "  # Bolus prévu (si applicable)
            f"{cas:>42} "                                        # Description du cas de fusion
            f"{'PANNE' if failed else 'OK':>8}"                  # État réel de la pompe
        )

        # ── Historiques ───────────────────────────────────────────────────────
        times.append(time_label)                                 # Enregistre le label de temps
        cgm_vals.append(cgm_val)                                 # Enregistre la valeur CGM
        pred_vals.append(pred)                                   # Enregistre la prédiction Kalman
        pump_states.append(failed)                               # Enregistre l'état de la pompe
        k_scores.append(k_score)                                 # Enregistre le score Kalman
        k_alerts_h.append(k_alert)                               # Enregistre l'alerte Kalman
        l_gfm_h.append(gfm)                                     # Enregistre la valeur GFM
        l_ifm_h.append(ifm)                                      # Enregistre la valeur IFM
        l_slope_h.append(slope)                                  # Enregistre la pente CGM
        l_consec_h.append(consec)                                # Enregistre le compteur consécutif LISA
        l_or_brut_h.append(or_brut)                              # Enregistre le signal OR brut
        l_alerts_h.append(l_alert)                               # Enregistre l'alerte LISA
        fusion_states.append(etat_fusion)                        # Enregistre l'état de fusion
        cas_history.append(cas)                                  # Enregistre le cas de fusion
        confiance_history.append(confiance)                      # Enregistre le niveau de confiance
        fusion_level_history.append(                             # Enregistre le niveau numérique de fusion
            2   if etat_fusion == "PANNE" else                   # 2 = PANNE
            1.5 if etat_fusion == "SUSPICION_SEQUENTIELLE" else  # 1.5 = SUSPICION SÉQUENTIELLE
            1   if etat_fusion == "SUSPICION" else 0             # 1 = SUSPICION, 0 = OK
        )
        real_level_history.append(3 if failed else 2.85)        # 3 si panne réelle, 2.85 si pompe OK

        # ── Suivi insuline ────────────────────────────────────────────────────
        if bolus_attendu_ce_step:                                # Si un bolus était prévu ce step
            bolus_total_prevu += 1                               # Incrémente le compteur de bolus prévus
            if failed:                                           # Si la pompe était en panne
                bolus_total_manque  += 1                         # Incrémente le compteur de bolus manqués
                insuline_manquee_U  += bolus_plan                # Ajoute la dose manquée au total
            else:                                                # Si la pompe fonctionnait
                bolus_total_injecte += 1                         # Incrémente le compteur de bolus injectés
                insuline_injectee_U += bolus_final               # Ajoute la dose injectée au total
        elif not failed and bolus_final > 0:                     # Si bolus de correction (pas de repas prévu)
            insuline_injectee_U += bolus_final                   # Ajoute le bolus de correction au total injecté
        if not failed:                                           # Si la pompe fonctionnait
            insuline_basale_U += dynamic_basal * sample_time     # Ajoute l'insuline basale délivrée ce step

        # ── Affichage graphique ───────────────────────────────────────────────
        disp_times         = times[-MAX_PTS:]                    # Fenêtre glissante : derniers MAX_PTS labels de temps
        disp_cgm           = cgm_vals[-MAX_PTS:]                 # Fenêtre glissante : dernières MAX_PTS valeurs CGM
        disp_pred          = pred_vals[-MAX_PTS:]                # Fenêtre glissante : dernières prédictions Kalman
        disp_states        = pump_states[-MAX_PTS:]              # Fenêtre glissante : derniers états de pompe
        disp_kscore        = k_scores[-MAX_PTS:]                 # Fenêtre glissante : derniers scores Kalman
        disp_gfm           = l_gfm_h[-MAX_PTS:]                 # Fenêtre glissante : dernières valeurs GFM
        disp_ifm           = l_ifm_h[-MAX_PTS:]                  # Fenêtre glissante : dernières valeurs IFM
        disp_slope         = l_slope_h[-MAX_PTS:]                # Fenêtre glissante : dernières pentes CGM
        disp_consec        = l_consec_h[-MAX_PTS:]               # Fenêtre glissante : derniers compteurs consécutifs
        disp_fus           = fusion_level_history[-MAX_PTS:]     # Fenêtre glissante : derniers niveaux de fusion
        disp_real          = real_level_history[-MAX_PTS:]       # Fenêtre glissante : derniers niveaux réels
        disp_fusion_states = fusion_states[-MAX_PTS:]            # Fenêtre glissante : derniers états de fusion
        disp_conf          = confiance_history[-MAX_PTS:]        # Fenêtre glissante : dernières confiances

        x      = list(range(len(disp_times)))                    # Indices X pour l'affichage (0 à len-1)
        offset = max(0, len(times) - MAX_PTS)                    # Décalage pour calculer les positions absolues

        line_cgm.set_data(x, disp_cgm)                          # Met à jour la courbe CGM
        line_pred.set_data(x, disp_pred)                         # Met à jour la courbe de prédiction Kalman
        line_kscore.set_data(x, disp_kscore)                     # Met à jour la courbe du score Kalman
        line_gfm.set_data(x, [v / 100.0 for v in disp_gfm])     # Met à jour la courbe GFM normalisée (/100)
        line_ifm.set_data(x, disp_ifm)                           # Met à jour la courbe IFM
        line_slope.set_data(x, disp_slope)                       # Met à jour la courbe de pente
        line_consec.set_data(x, [v / max(1, LISA_CONFIRMATION_STEPS) for v in disp_consec])  # Met à jour la courbe consécutifs normalisée

        if fill_ref[0] is not None:                              # Si une zone remplie existe déjà
            fill_ref[0].remove()                                 # Supprime l'ancienne zone (pour éviter l'accumulation)
        fill_ref[0] = ax_cgm.fill_between(x, disp_cgm, alpha=0.08, color=C_CGM, zorder=2)  # Recrée la zone sous la courbe CGM

        # Panneau fusion — barres colorées
        ax_fus.cla()                                             # Efface le panneau de fusion pour le redessiner
        ax_fus.set_facecolor(C_BG)                               # Remet le fond blanc
        ax_fus.grid(True, color=C_GRID, alpha=0.6, linewidth=0.5)  # Remet la grille
        ax_fus.set_ylim(-0.5, 3.5)                               # Remet les limites Y
        ax_fus.set_yticks([0, 1, 1.5, 2, 3])                     # Remet les graduations Y
        ax_fus.set_yticklabels(["OK", "SUSP.", "SUSP.\nSEQ.", "PANNE\nfusion", "PANNE\nréelle"],  # Labels des états
                                color=C_TICK, fontsize=6)
        ax_fus.set_ylabel("Décision", color=C_TICK, fontsize=7)  # Label de l'axe Y
        ax_fus.set_xlabel("Heure simulée", color=C_TICK, fontsize=8)  # Label de l'axe X
        ax_fus.tick_params(colors=C_TICK, labelsize=7)           # Style des graduations
        for s_name in ["top", "right"]:   ax_fus.spines[s_name].set_visible(False)  # Cache les bordures haut/droite
        for s_name in ["bottom", "left"]: ax_fus.spines[s_name].set_color(C_SPINE)  # Colore les bordures bas/gauche

        for i_f, (fv, rv, fc) in enumerate(zip(disp_fus, disp_real, disp_conf)):  # Boucle sur chaque step affiché
            if fv == 2:     color_f = C_HAUTE                    # Panne confirmée → rouge
            elif fv == 1.5: color_f = C_SEQ                     # Suspicion séquentielle → or foncé
            elif fv == 1:   color_f = "#e6a817"                  # Suspicion simple → orange
            else:           color_f = "#c8e6c9"                  # OK → vert clair
            ax_fus.bar(i_f, fv + 0.5, bottom=-0.5, width=1.0,   # Dessine la barre de hauteur proportionnelle à l'état
                       color=color_f, alpha=0.80, align="edge")
            ax_fus.plot(i_f + 0.5, rv, marker="_", ms=10,        # Dessine le marqueur de l'état réel de la pompe
                        color=C_BOLUS_FAIL, lw=2, alpha=0.9)

        for ax in AXES:                                          # Met à jour les limites X de tous les axes
            ax.set_xlim(-1, len(disp_times))                     # X de -1 à len pour une marge visuelle

        ticks = list(range(0, len(disp_times), 6))               # Crée les positions des graduations X (tous les 6 steps)
        ax_fus.set_xticks(ticks)                                  # Applique les graduations X au panneau fusion
        ax_fus.set_xticklabels(                                  # Applique les labels horaires correspondants
            [disp_times[i] for i in ticks if i < len(disp_times)],
            fontsize=7, color="#64748b"
        )
        for ax in [ax_cgm, ax_k, ax_l]:                          # Pour les autres panneaux
            ax.set_xticks(ticks)                                  # Applique les mêmes graduations
            ax.set_xticklabels([], visible=False)                 # Mais cache les labels (seul le panneau fusion les affiche)

        # Bandes panne/marche
        for ax in AXES:                                          # Pour chaque axe graphique
            for band in failure_bands[ax]:                       # Supprime les anciennes bandes de fond
                try: band.remove()
                except Exception: pass
            failure_bands[ax].clear()                            # Vide la liste des bandes de cet axe

        i_seg = 0                                                # Indice de début du segment courant
        while i_seg < len(disp_states):                          # Parcourt tous les états de pompe affichés
            seg_state = disp_states[i_seg]; j_seg = i_seg        # Initialise le segment avec l'état courant
            while j_seg < len(disp_states) and disp_states[j_seg] == seg_state:  # Avance tant que l'état ne change pas
                j_seg += 1
            c_band = C_BAND_FAIL if seg_state else C_BAND_OK     # Choisit la couleur selon l'état (rouge=panne, vert=OK)
            a_band = 0.45 if seg_state else 0.35                 # Opacité plus forte pour les pannes
            for ax in AXES:                                      # Dessine la bande colorée sur tous les axes
                b = ax.axvspan(i_seg - 0.5, j_seg - 0.5, alpha=a_band, color=c_band, zorder=1)
                failure_bands[ax].append(b)                      # Mémorise la bande pour pouvoir la supprimer au prochain step
            i_seg = j_seg                                        # Avance à l'état suivant

        # Repas
        mx, my = [], []                                          # Listes X et Y pour les marqueurs de repas
        for r in scenario.repas:                                 # Parcourt tous les repas du scénario
            r_step = r.get("step")                               # Récupère le step du repas
            if r_step is None: continue                          # Ignore les repas sans step défini
            vi = r_step - offset                                 # Calcule la position X dans la fenêtre affichée
            if 0 <= vi < len(disp_times):                        # Si le repas est dans la fenêtre affichée
                mx.append(vi); my.append(disp_cgm[vi] + 18)     # Positionne le marqueur légèrement au-dessus du CGM
        sc_meals.set_offsets(np.c_[mx, my] if mx else np.empty((0, 2)))  # Met à jour les marqueurs de repas

        # Bolus
        ok_x, ok_y, fx, fy = [], [], [], []                      # Listes pour bolus injectés (ok) et manqués (f)
        for step_b in BOLUS_STEPS:                               # Parcourt tous les steps de bolus prévus
            vi = step_b - offset                                 # Position X dans la fenêtre affichée
            if 0 <= vi < len(disp_times):                        # Si dans la fenêtre
                if pump_is_failed(step_b): fx.append(vi); fy.append(disp_cgm[vi] - 18)    # Bolus manqué (croix en dessous)
                else:                      ok_x.append(vi); ok_y.append(disp_cgm[vi] - 18)  # Bolus injecté (cercle en dessous)
        sc_bolus_ok.set_offsets(np.c_[ok_x, ok_y] if ok_x else np.empty((0, 2)))   # Met à jour les cercles de bolus OK
        sc_bolus_fail.set_offsets(np.c_[fx, fy] if fx else np.empty((0, 2)))        # Met à jour les croix de bolus manqués

        # Alertes
        ph_x, ph_y, su_x, su_y, sq_x, sq_y = [], [], [], [], [], []  # Listes pour les différents types d'alerte
        for i_a, (fs, fc) in enumerate(zip(disp_fusion_states, disp_conf)):  # Parcourt les états de fusion affichés
            if fs == "PANNE":                                    # Si panne confirmée
                ph_x.append(i_a); ph_y.append(disp_cgm[i_a] + 30)  # Marqueur diamant au-dessus du CGM (+30 mg/dL)
            elif fs == "SUSPICION_SEQUENTIELLE":                 # Si suspicion séquentielle
                sq_x.append(i_a); sq_y.append(disp_cgm[i_a] + 25)  # Marqueur carré au-dessus du CGM (+25 mg/dL)
            elif fs == "SUSPICION":                              # Si suspicion simple
                su_x.append(i_a); su_y.append(disp_cgm[i_a] + 20)  # Marqueur cercle au-dessus du CGM (+20 mg/dL)
        sc_panne_haute.set_offsets(np.c_[ph_x, ph_y] if ph_x else np.empty((0, 2)))  # Met à jour les diamants de panne
        sc_susp_seq.set_offsets(np.c_[sq_x, sq_y] if sq_x else np.empty((0, 2)))     # Met à jour les carrés de suspicion séq.
        sc_suspicion.set_offsets(np.c_[su_x, su_y] if su_x else np.empty((0, 2)))    # Met à jour les cercles de suspicion

        # Textes
        if etat_fusion == "PANNE":                     col_etat = C_HAUTE   # Couleur rouge pour panne
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":  col_etat = C_SEQ     # Couleur or pour suspicion séquentielle
        elif etat_fusion == "SUSPICION":               col_etat = "#e6a817" # Couleur orange pour suspicion simple
        else:                                          col_etat = C_CGM     # Couleur CGM (vert-bleu) pour état OK
        txt_etat.set_text(f"{cgm_val:.0f} mg/dL  [{etat_fusion}]")          # Affiche glycémie et état de fusion
        txt_etat.set_color(col_etat)                                         # Applique la couleur selon l'état

        txt_info.set_text(                                       # Met à jour le texte d'informations détaillées
            f"{time_label} | step {index} | "                    # Heure et numéro de step
            f"Kalman {'✓' if k_alert else '·'}  LISA {'✓' if l_alert else '·'} "  # Indicateurs d'alerte K et L
            f"(Score≥{LISA_SCORE_MIN}: {'✓' if or_brut else '·'} ×{consec}/{LISA_CONFIRMATION_STEPS})  |  "  # Score LISA
            f"GFM={gfm:.1f}  IFM={ifm:.3f}  Pente={slope:.2f}  |  "       # Valeurs des indicateurs LISA
            f"Réel : {'PANNE' if failed else 'OK'}"                         # État réel de la pompe
        )

        ecart_txt = ""                                           # Texte décrivant l'écart entre alertes (vide par défaut)
        if fusion.ecarts:                                        # S'il y a des écarts enregistrés
            last_step, last_ecart, last_conf = fusion.ecarts[-1] # Récupère le dernier écart enregistré
            if last_ecart is not None:                           # Si l'écart est défini (deux détecteurs ont alerté)
                signe     = "+" if last_ecart > 0 else ""        # Détermine le signe de l'écart
                ecart_txt = (f"Écart K→L : {signe}{last_ecart} steps "  # Affiche l'écart en steps
                             f"({signe}{last_ecart*sample_time:.0f} min)")  # Et en minutes
            else:                                                # Si un seul détecteur a alerté
                ecart_txt = f"Un seul détecteur ({last_conf})"   # Affiche le niveau de confiance du détecteur seul
        txt_cas.set_text(f"Cas : {cas}\nConfiance : {confiance if confiance else '—'}\n{ecart_txt}")  # Met à jour le texte de cas

        if etat_fusion == "PANNE":                               # Mise à jour du texte d'alarme central selon l'état
            txt_alarm.set_text("PANNE CONFIRMÉE — Kalman + LISA")  # Message de panne confirmée
            txt_alarm.set_color(C_HAUTE)                         # Texte rouge
            txt_alarm.get_bbox_patch().set_edgecolor(C_HAUTE)    # Bordure rouge
            txt_alarm.get_bbox_patch().set_facecolor("#ffe5e5")  # Fond rose clair
            txt_alarm.get_bbox_patch().set_alpha(0.90)           # Opacité 90%
        elif etat_fusion == "SUSPICION_SEQUENTIELLE":            # Si suspicion séquentielle
            if (fusion._k_first_alert_step is not None and       # Détermine l'ordre des alertes K et L
                    fusion._l_first_alert_step is not None):
                _ordre = "K → L" if fusion._k_first_alert_step <= fusion._l_first_alert_step else "L → K"  # Ordre chronologique
            else:
                _ordre = "K ↔ L"                                 # Ordre indéterminé
            txt_alarm.set_text(f"SUSPICION SÉQUENTIELLE — {_ordre}")  # Message avec l'ordre des alertes
            txt_alarm.set_color(C_SEQ)                           # Texte or foncé
            txt_alarm.get_bbox_patch().set_edgecolor(C_SEQ)      # Bordure or
            txt_alarm.get_bbox_patch().set_facecolor("#fff3cd")  # Fond jaune clair
            txt_alarm.get_bbox_patch().set_alpha(0.90)           # Opacité 90%
        elif etat_fusion == "SUSPICION":                         # Si suspicion simple
            txt_alarm.set_text(f"SUSPICION — {detecteur_txt}")   # Message avec le nom du détecteur responsable
            txt_alarm.set_color("#7a5c00")                       # Texte brun foncé
            txt_alarm.get_bbox_patch().set_edgecolor("#e6a817")  # Bordure orange
            txt_alarm.get_bbox_patch().set_facecolor("#fff8e1")  # Fond jaune très clair
            txt_alarm.get_bbox_patch().set_alpha(0.85)           # Opacité 85%
        else:                                                    # Si état OK
            txt_alarm.set_text("")                               # Efface le message d'alarme
            txt_alarm.get_bbox_patch().set_alpha(0.0)            # Rend la boîte invisible

        try:
            fig.canvas.draw_idle()                               # Demande un redessin asynchrone du graphique
            fig.canvas.flush_events()                            # Traite les événements GUI en attente
        except Exception:
            pass                                                 # Ignore les erreurs de rendu (fenêtre fermée, etc.)

        time.sleep(STEP_DELAY)                                   # Pause entre steps pour rendre la simulation visible

        if done:                                                 # Si l'épisode de simulation est terminé
            print(f"\n[INFO] Reset environnement (step {index})\n")  # Informe de la réinitialisation
            p2  = T1DPatient.withName(PATIENT_NAME)              # Recrée le patient pour le nouvel épisode
            s2  = CGMSensor.withName("Dexcom", seed=index)       # Recrée le capteur avec une nouvelle graine
            pu2 = InsulinPump.withName("Insulet")                # Recrée la pompe
            sc2 = scenario                                       # Réutilise le scénario existant
            sc2.start_time = start_time + timedelta(minutes=index * sample_time)  # Ajuste le temps de départ du scénario
            env2 = T1DSimEnv(patient=p2, sensor=s2, pump=pu2, scenario=sc2)      # Crée le nouvel environnement
            s    = env2.reset()                                  # Réinitialise et récupère l'observation initiale
            env  = env2                                          # Remplace l'environnement courant
            kalman = KalmanDetector(float(s.observation.CGM), sample_time)  # Réinitialise Kalman avec le nouveau CGM
            lisa   = LISADetector(sample_time)                   # Réinitialise LISA
        else:
            s = env.step(action)                                 # Avance l'environnement d'un step avec l'action courante

        obs  = s.observation                                     # Récupère la nouvelle observation
        done = s.done                                            # Récupère l'indicateur de fin d'épisode
        index += 1                                               # Incrémente le compteur de steps


except KeyboardInterrupt:                                        # Capture l'interruption Ctrl+C
    print(f"\n[INFO] Simulation arrêtée par l'utilisateur au step {index}.")  # Affiche le step final à l'arrêt

plt.ioff()                                                       # Désactive le mode interactif matplotlib à la fin