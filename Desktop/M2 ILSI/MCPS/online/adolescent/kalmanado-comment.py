# =============================================================================
# kalmanado_commente.py
# Calibration offline d'un filtre de Kalman "gelé" pour détecter les bolus
# manqués chez un patient ADOLESCENT — simulation sur 40 heures
# =============================================================================

import numpy as np
from datetime import datetime
from collections import namedtuple

# Imports du simulateur de glycémie (simglucose)
from simglucose.simulation.env import T1DSimEnv          # Environnement de simulation
from simglucose.simulation.scenario import Scenario       # Classe de base pour les scénarios
from simglucose.simulation.scenario import Action as MealAction  # Action repas (glucides)
from simglucose.patient.t1dpatient import T1DPatient      # Modèle du patient diabétique
from simglucose.sensor.cgm import CGMSensor               # Capteur de glycémie (CGM)
from simglucose.actuator.pump import InsulinPump          # Pompe à insuline

# =============================================================================
# IDENTITÉ DU PATIENT
# =============================================================================
PATIENT_NAME = "adolescent#001"  # Nom du patient dans la base simglucose

# =============================================================================
# PLANNING DES REPAS SUR UNE JOURNÉE
# Chaque repas est défini par :
#   - min_in_day : minute de la journée simulée où le repas a lieu
#   - carbs      : glucides en grammes
#   - label      : nom du repas (informatif seulement)
# =============================================================================
REPAS_JOURNEE = [
    {"min_in_day":  60, "carbs": 45, "label": "Petit-dej 45g"},
    {"min_in_day": 180, "carbs": 60, "label": "Dejeuner 60g"},
    {"min_in_day": 300, "carbs": 20, "label": "Gouter 20g"},
    {"min_in_day": 420, "carbs": 55, "label": "Diner 55g"},
    {"min_in_day": 540, "carbs": 15, "label": "Collation 15g"},
]

# =============================================================================
# PARAMÈTRES INSULINE
# =============================================================================
ICR  = 15.0   # Insulin-to-Carb Ratio : 1 unité couvre 15g de glucides
ISF  = 60.0   # Insulin Sensitivity Factor : 1 unité baisse la glycémie de 60 mg/dL
CGM_CIBLE = 120.0  # Glycémie cible (mg/dL) — valeur idéale visée
BOLUS_MIN = 0.3    # Dose bolus minimale injectée (unités) — sécurité basse
BOLUS_MAX = 10.0   # Dose bolus maximale autorisée (unités) — sécurité haute

# =============================================================================
# PARAMÈTRES DE SIMULATION
# =============================================================================
BASAL             = 0.040  # Dose d'insuline basale injectée à chaque pas (unités/min)
PREBOLUS_STEPS    = 8      # Le bolus est injecté 8 pas AVANT le repas (anticipation)
PANNE_DUREE       = 32     # Durée d'une panne de pompe (en pas de simulation)
MARCHE_DUREE      = 20     # Durée de fonctionnement normal entre deux pannes (en pas)
PANNE_START_STEP  = 10     # Pas à partir duquel les pannes commencent
MINUTES_PAR_JOUR  = 600    # Durée d'une journée simulée (600 min = 10 heures)
MAX_STEPS         = 800    # Nombre total de pas — 800 × 3 min = 2400 min = 40 heures

# =============================================================================
# PARAMÈTRES DU FILTRE DE KALMAN
# Q = confiance dans le modèle (fort Q = on veut corriger vite la glycémie)
# R = prudence sur les mesures (fort R = on injecte peu d'un coup)
# =============================================================================
Q_PROFILE = np.array([[1.2, 0.0], [0.0, 0.12]])  # Matrice de bruit de processus
R_PROFILE = np.array([[25.0]])                     # Matrice de bruit de mesure

# =============================================================================
# GRILLES DE RECHERCHE (grid search) — valeurs testées pour trouver le meilleur
# jeu de paramètres de détection
# =============================================================================
FENETRES           = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]  # Tailles de fenêtres (en pas)
PERCENTILES        = [85, 88, 90, 92, 94, 95, 96, 97, 98, 99] # Seuils statistiques testés
CONFIRM_STEPS_LIST = [1, 2, 3, 4]                              # Nombre de pas consécutifs pour confirmer une alarme
SLOPE_MIN_LIST     = [0.0, 0.2, 0.5, 1.0]                     # Pentes minimales (mg/dL/min) pour valider une montée
SLOPE_LOOKBACK_STEPS = 3  # Nombre de pas en arrière pour calculer la pente courante

# =============================================================================
# POIDS DE LA FONCTION OBJECTIF
# La calibration choisit les paramètres qui minimisent ce score :
#   score = W_FN × faux_negatifs + W_FP × faux_positifs + W_DELAY × délai
# FN (rater un bolus manqué) est 3× plus grave qu'un FP (fausse alarme)
# =============================================================================
W_FN  = 3.0    # Poids des faux négatifs (bolus manqué non détecté)
W_FP  = 1.0    # Poids des faux positifs (fausse alarme)
W_DELAY = 0.02 # Pénalité par minute de retard dans la détection
DELAI_MAX_ACCEPTABLE = 180.0  # Délai max toléré (min) — au-delà c'est inacceptable

# =============================================================================
# STRUCTURE DE DONNÉES
# Action regroupe les deux sorties de la pompe à chaque pas
# =============================================================================
Action = namedtuple("Action", ["basal", "bolus"])
# Exemple d'usage : Action(basal=0.040, bolus=3.0)


# =============================================================================
# GÉNÉRATION DES REPAS SUR TOUTE LA SIMULATION
# =============================================================================
def build_repas(sample_time, max_steps):
    """
    Construit la liste complète de tous les repas sur la durée de la simulation.

    Args:
        sample_time : durée d'un pas en minutes (ex : 3 min)
        max_steps   : nombre total de pas de simulation

    Returns:
        repas : liste de dicts {"min": ..., "carbs": ...}
                chaque dict représente un repas avec sa minute absolue
    """
    repas = []

    # Calcul du nombre de jours nécessaires pour couvrir toute la simulation
    # +2 pour s'assurer qu'on ne coupe pas le dernier jour
    total_min = max_steps * sample_time
    nb_jours  = int(total_min / MINUTES_PAR_JOUR) + 2

    for jour in range(nb_jours):
        base_min = jour * MINUTES_PAR_JOUR  # Minute de départ du jour courant

        for r in REPAS_JOURNEE:
            # On calcule la minute absolue du repas dans la simulation
            repas.append({
                "min":   base_min + r["min_in_day"],
                "carbs": r["carbs"],
            })

    return repas


# =============================================================================
# CALCUL DU BOLUS POUR UN REPAS
# =============================================================================
def calculer_bolus(carbs, cgm_val):
    """
    Calcule la dose de bolus insuline à injecter pour un repas.

    Logique :
      1. Si glycémie < 70 → hypoglycémie, on n'injecte rien (trop dangereux)
      2. Bolus repas = glucides / ICR  (couvre les glucides du repas)
      3. Bolus correction = (glycémie - cible) / ISF  (corrige si glycémie trop haute)
      4. Total clampé entre BOLUS_MIN et BOLUS_MAX pour la sécurité

    Args:
        carbs   : glucides du repas (grammes)
        cgm_val : glycémie actuelle lue par le capteur (mg/dL)

    Returns:
        bolus total arrondi à 2 décimales (unités d'insuline)
    """
    # Sécurité : pas d'insuline si le patient est déjà en hypoglycémie
    if cgm_val < 70.0:
        return 0.0

    # Bolus pour couvrir les glucides du repas
    bolus_repas = carbs / ICR

    # Bolus de correction si la glycémie est au-dessus de 80 mg/dL
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0

    # On additionne les deux et on borne entre BOLUS_MIN et BOLUS_MAX
    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)


# =============================================================================
# SCÉNARIO PATIENT — gère les repas dans le simulateur
# =============================================================================
class PatientScenario(Scenario):
    """
    Scénario de simulation qui reproduit le comportement alimentaire du patient.
    À chaque pas de temps, le simulateur appelle get_action() pour savoir
    si le patient mange quelque chose à ce moment-là.
    """

    def __init__(self, start_time, repas, sample_time):
        """
        Args:
            start_time  : datetime de début de la simulation
            repas       : liste de repas avec minute absolue et glucides
            sample_time : durée d'un pas en minutes
        """
        self.start_time  = start_time
        self.sample_time = sample_time

        # Dictionnaire : numéro_de_pas → glucides
        # Permet de retrouver instantanément si un repas a lieu à un pas donné
        self.meal_carbs = {
            int(r["min"] / sample_time): r["carbs"]
            for r in repas
        }

    def get_action(self, t):
        """
        Appelé par le simulateur à chaque pas de temps t.
        Retourne les glucides ingérés à ce moment (0 si pas de repas).

        Args:
            t : datetime du pas courant

        Returns:
            MealAction avec le nombre de grammes de glucides (0 si pas de repas)
        """
        # Calcul du temps écoulé depuis le début (en minutes)
        elapsed = (t - self.start_time).total_seconds() / 60

        # Conversion en numéro de pas
        step = int(round(elapsed / self.sample_time))

        # Retourne les glucides du repas, ou 0 s'il n'y a pas de repas à ce pas
        return MealAction(meal=self.meal_carbs.get(step, 0))

    def reset(self):
        pass  # Rien à réinitialiser, le planning est fixe


# =============================================================================
# DÉTECTION DE PANNE DE POMPE
# La pompe alterne entre phases de panne et de fonctionnement normal
# =============================================================================
def pump_is_failed(step):
    """
    Indique si la pompe est en panne au pas de simulation 'step'.

    Logique :
      - Avant PANNE_START_STEP  → pompe toujours OK
      - Après PANNE_START_STEP  → alternance panne / marche cyclique :
            [panne : PANNE_DUREE pas] [marche : MARCHE_DUREE pas] [panne...] ...

    Args:
        step : numéro du pas courant

    Returns:
        True si la pompe est en panne, False sinon
    """
    if step < PANNE_START_STEP:
        return False  # Pas encore de panne

    # Position dans le cycle courant (panne + marche)
    position_dans_cycle = (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE)

    # On est en panne si la position est dans la première partie du cycle
    return position_dans_cycle < PANNE_DUREE


# =============================================================================
# CONTRÔLEUR INSULINE BASALE (simple, à base de règles)
# =============================================================================
def insulin_controller(cgm_val, bolus_plan):
    """
    Détermine la dose basale à injecter en fonction de la glycémie actuelle.
    Ajuste la basale à la hausse si hyperglycémie, à la baisse si hypoglycémie.

    Args:
        cgm_val    : glycémie actuelle (mg/dL)
        bolus_plan : bolus déjà prévu (transmis tel quel en sortie)

    Returns:
        (basal_ajusté, bolus_plan) : tuple des doses à injecter
    """
    db = BASAL  # On part de la dose basale standard

    # Augmentation si hyperglycémie légère (>170) ou sévère (>210)
    if cgm_val > 170: db += 0.007
    if cgm_val > 210: db += 0.012

    # Réduction si hypoglycémie légère (<100) ou sévère (<85)
    if cgm_val < 100: db -= 0.007
    if cgm_val < 85:  db -= 0.012

    # Coupure totale si hypoglycémie franche (<70) — sécurité absolue
    if cgm_val < 70:  db = 0.0

    return max(0.0, db), bolus_plan  # On garantit une dose ≥ 0


# =============================================================================
# SIMULATION COMPLÈTE
# =============================================================================
def run_simulation(with_fault=False):
    """
    Lance la simulation sur MAX_STEPS pas.

    Args:
        with_fault : si True → les pannes de pompe sont activées

    Returns:
        cgm_values          : tableau numpy des glycémies à chaque pas
        pump_failed_labels  : tableau bool — True si pompe en panne au pas i
        missed_bolus_labels : tableau bool — True si un bolus a été raté au pas i
        bolus_steps_sorted  : liste triée des pas où un bolus devait être injecté
        sample_time         : durée d'un pas en minutes (récupérée du simulateur)
    """
    start_time = datetime.now()

    # --- Récupération du sample_time réel du simulateur (en minutes) ---
    # On crée un environnement temporaire juste pour lire sample_time
    env_tmp = T1DSimEnv(
        patient=T1DPatient.withName(PATIENT_NAME),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=PatientScenario(start_time, [{"min": 60, "carbs": 30}], 3),
    )
    sample_time = env_tmp.sample_time  # Ex : 3 minutes par pas
    del env_tmp  # On supprime l'environnement temporaire

    # --- Création de l'environnement réel ---
    repas = build_repas(sample_time, MAX_STEPS)  # Génère tous les repas
    env = T1DSimEnv(
        patient=T1DPatient.withName(PATIENT_NAME),
        sensor=CGMSensor.withName("Dexcom", seed=1),
        pump=InsulinPump.withName("Insulet"),
        scenario=PatientScenario(start_time, repas, sample_time),
    )

    # --- Calcul des pas où les bolus doivent être injectés ---
    # Le bolus est injecté PREBOLUS_STEPS pas AVANT le repas
    bolus_steps_map = {}
    for r in repas:
        step_repas = int(r["min"] / sample_time)                    # Pas du repas
        step_bolus = max(0, step_repas - PREBOLUS_STEPS)            # Pas du bolus (anticipé)
        bolus_steps_map[step_bolus] = r["carbs"]                    # Glucides prévus à ce pas
    bolus_steps_sorted = sorted(bolus_steps_map.keys())             # Liste triée des pas-bolus

    # --- Boucle principale de simulation ---
    s = env.reset()       # Initialisation, retourne le premier état
    obs = s.observation   # Observation initiale (contient la glycémie CGM)

    cgm_values          = []  # Historique des glycémies
    pump_failed_labels  = []  # Historique des états de la pompe (True = en panne)
    missed_bolus_labels = []  # Historique des bolus manqués (True = bolus raté)

    for step in range(MAX_STEPS):
        cgm_val   = float(obs.CGM)  # Lecture de la glycémie au pas courant

        # Est-ce qu'un bolus est prévu à ce pas ?
        carbs_now  = bolus_steps_map.get(step, 0)
        bolus_plan = calculer_bolus(carbs_now, cgm_val) if carbs_now > 0 else 0.0

        # État de la pompe au pas courant
        failed = pump_is_failed(step) if with_fault else False

        # Un bolus est "manqué" si la pompe est en panne ET qu'un bolus était prévu
        missed = failed and (carbs_now > 0)

        # Injection ou coupure selon l'état de la pompe
        if failed:
            action = Action(basal=0.0, bolus=0.0)  # Pompe en panne → rien injecté
        else:
            db, tb = insulin_controller(cgm_val, bolus_plan)
            action = Action(basal=db, bolus=tb)     # Pompe OK → injection normale

        # Enregistrement de ce pas
        cgm_values.append(cgm_val)
        pump_failed_labels.append(failed)
        missed_bolus_labels.append(missed)

        # Avancer d'un pas dans le simulateur
        s   = env.step(action)
        obs = s.observation
        if s.done:
            break  # Fin anticipée si le simulateur signale la fin

    return (
        np.array(cgm_values),
        np.array(pump_failed_labels),
        np.array(missed_bolus_labels),
        bolus_steps_sorted,
        sample_time,
    )


# =============================================================================
# PARAMÈTRE DU DÉGEL
# Le filtre Kalman gelé se dégèle après N_DEGEL pas consécutifs de descente
# =============================================================================
N_DEGEL = 3


# =============================================================================
# FILTRE DE KALMAN GELÉ — CALCUL DES SCORES D'ANOMALIE
# =============================================================================
def kalman_frozen_scores(cgm_values, sample_time, window_size):
    """
    Applique un filtre de Kalman "gelé" sur les glycémies pour produire
    un score d'anomalie à chaque pas.

    Principe :
      - En mode LIBRE  : le filtre suit normalement la glycémie → score = 0
      - En mode GELÉ   : la prédiction est figée, on mesure l'écart croissant
                         → score = moyenne des écarts sur la fenêtre
      - Gel déclenché quand glycémie > prédiction (montée anormale)
      - Dégel déclenché après N_DEGEL pas consécutifs de descente

    Args:
        cgm_values  : tableau des glycémies (mg/dL)
        sample_time : durée d'un pas en minutes
        window_size : taille de la fenêtre glissante pour moyenner les écarts

    Returns:
        scores : tableau numpy — score à chaque pas (0 = normal, >0 = suspect)
    """
    dt = sample_time  # Durée d'un pas (utilisée dans la matrice de transition)

    # --- Matrices du filtre de Kalman ---
    # F : matrice de transition d'état
    #     [glycémie_suivante] = [glycémie] + dt × [pente]
    #     [pente_suivante   ] = [pente]
    F = np.array([[1, dt], [0, 1]])

    # H : matrice d'observation — on ne mesure que la glycémie (pas la pente)
    H = np.array([[1, 0]])

    # x : état initial — [glycémie initiale, pente = 0]
    x = np.array([[cgm_values[0]], [0.0]])

    # P : matrice de covariance initiale — incertitude de départ
    P = np.eye(2) * 10.0

    # --- Variables internes ---
    ecarts_fenetre = []   # Fenêtre glissante des écarts accumulés en mode gelé
    scores         = []   # Score de chaque pas (résultat final)
    frozen         = False  # Mode courant : False = libre, True = gelé
    neg_slope_count = 0     # Compteur de pas consécutifs en descente (pour le dégel)
    prev_cgm        = cgm_values[0]  # Glycémie du pas précédent

    for cgm in cgm_values:

        # --- Étape de prédiction (commune aux deux modes) ---
        x_pred = F @ x           # Prédiction de l'état au pas suivant
        P_pred = F @ P @ F.T + Q_PROFILE  # Prédiction de l'incertitude

        # Glycémie prédite par le filtre (valeur libre, non corrigée)
        p_libre = float((H @ x_pred)[0, 0])

        # Écart entre la vraie glycémie et la prédiction du filtre
        ecart = cgm - p_libre

        # --- Comptage des pas en descente (pour décider du dégel) ---
        if cgm - prev_cgm < 0:
            neg_slope_count += 1  # La glycémie baisse → on incrémente
        else:
            neg_slope_count = 0   # La glycémie remonte → on remet à zéro

        # --- Dégel : si en mode gelé et assez de pas en descente ---
        if frozen and neg_slope_count >= N_DEGEL:
            frozen = False           # On sort du mode gelé
            ecarts_fenetre = []      # On efface la fenêtre d'écarts

            # On remet le filtre à jour avec la vraie valeur (correction Kalman)
            z        = np.array([[cgm]])
            residual = z - H @ x_pred
            S = H @ P_pred @ H.T + R_PROFILE
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ residual
            P = (np.eye(2) - K @ H) @ P_pred

            scores.append(0.0)   # Score nul au pas de dégel
            prev_cgm = cgm
            continue             # On passe au pas suivant

        # --- Déclenchement du gel : glycémie > prédiction et on est libre ---
        if ecart > 0 and not frozen:
            frozen = True        # On gèle le filtre
            neg_slope_count = 0  # On repart à zéro pour le compteur de descente

        # === MODE GELÉ ===
        if frozen:
            # On accumule l'écart dans la fenêtre glissante
            ecarts_fenetre.append(abs(ecart))

            # On limite la fenêtre à window_size éléments
            if len(ecarts_fenetre) > window_size:
                ecarts_fenetre.pop(0)  # On supprime le plus ancien

            # Le score est la moyenne des écarts sur la fenêtre
            # Plus il est élevé, plus la dérive est importante
            score = float(np.mean(ecarts_fenetre))

            # On NE met PAS à jour x et P — c'est tout l'intérêt du gel
            x, P = x_pred, P_pred

        # === MODE LIBRE ===
        else:
            ecarts_fenetre = []  # On efface les écarts accumulés

            # Correction classique du filtre de Kalman
            z        = np.array([[cgm]])
            residual = z - H @ x_pred              # Résidu = mesure - prédiction
            S = H @ P_pred @ H.T + R_PROFILE       # Covariance du résidu
            K = P_pred @ H.T @ np.linalg.inv(S)   # Gain de Kalman
            x = x_pred + K @ residual              # Mise à jour de l'état
            P = (np.eye(2) - K @ H) @ P_pred       # Mise à jour de l'incertitude

            score = 0.0  # Pas d'anomalie en mode libre

        scores.append(score)
        prev_cgm = cgm

    return np.array(scores)


# =============================================================================
# MASQUE DE PENTE — filtre les signaux dont la pente est trop faible
# =============================================================================
def compute_slope_mask(cgm_values, sample_time, slope_min):
    """
    Calcule pour chaque pas si la pente de la glycémie dépasse slope_min.
    Sert à filtrer les alarmes déclenchées sans vraie montée.

    Args:
        cgm_values : tableau des glycémies
        sample_time: durée d'un pas (minutes)
        slope_min  : pente minimale requise (mg/dL/min) ; 0 = juste positive

    Returns:
        mask : tableau bool — True si la pente est suffisante au pas i
    """
    mask = np.zeros(len(cgm_values), dtype=bool)

    for i in range(SLOPE_LOOKBACK_STEPS, len(cgm_values)):
        # Pente calculée sur les SLOPE_LOOKBACK_STEPS derniers pas
        slope = (cgm_values[i] - cgm_values[i - SLOPE_LOOKBACK_STEPS]) / (SLOPE_LOOKBACK_STEPS * sample_time)

        if slope_min <= 0:
            mask[i] = slope > 0.0       # On accepte toute pente positive
        else:
            mask[i] = slope >= slope_min  # On n'accepte que les pentes ≥ slope_min

    return mask


# =============================================================================
# CONFIRMATION D'ALARME — évite les détections isolées (bruit)
# =============================================================================
def apply_confirmation(raw_alarm, confirm_steps):
    """
    Transforme un signal brut d'alarme en alarme confirmée.
    Une alarme n'est validée que si elle est active pendant confirm_steps
    pas CONSÉCUTIFS.

    Args:
        raw_alarm     : tableau bool — alarme brute à chaque pas
        confirm_steps : nombre de pas consécutifs requis pour confirmer

    Returns:
        alarm : tableau bool — alarme confirmée (True seulement après confirmation)
    """
    alarm  = np.zeros(len(raw_alarm), dtype=bool)
    consec = 0  # Compteur de pas consécutifs actifs

    for i, active in enumerate(raw_alarm):
        consec = consec + 1 if active else 0  # Incrément ou remise à zéro
        if consec >= confirm_steps:
            alarm[i] = True  # Alarme confirmée après assez de pas consécutifs

    return alarm


# =============================================================================
# FENÊTRES DE DÉTECTION — une fenêtre par bolus manqué
# =============================================================================
def build_detection_windows(missed_bolus_labels, bolus_steps_sorted, n_steps):
    """
    Pour chaque bolus manqué, construit la fenêtre de temps pendant laquelle
    une détection est attendue (entre le bolus manqué et le bolus suivant).

    Args:
        missed_bolus_labels : tableau bool des bolus manqués
        bolus_steps_sorted  : pas triés où des bolus étaient prévus
        n_steps             : nombre total de pas de simulation

    Returns:
        windows : liste de tuples (ws, we) — pas de début et de fin de chaque fenêtre
    """
    # On garde uniquement les pas où un bolus a vraiment été manqué
    missed_steps = [s for s in bolus_steps_sorted if s < n_steps and missed_bolus_labels[s]]

    windows = []
    for ms in missed_steps:
        # La fenêtre se termine au pas juste avant le bolus suivant
        next_bolus = [s for s in bolus_steps_sorted if s > ms]
        we = next_bolus[0] - 1 if next_bolus else n_steps - 1
        windows.append((ms, we))

    return windows


# =============================================================================
# CALCUL DES MÉTRIQUES DE DÉTECTION
# =============================================================================
def compute_metrics(alarm_normal, alarm_fault, missed_bolus_labels, bolus_steps_sorted, sample_time):
    """
    Évalue la qualité de la détection par rapport à la vérité terrain.
    Les métriques sont calculées fenêtre par fenêtre (inter-bolus).

    Définitions :
      VP (Vrai Positif)  : bolus manqué détecté correctement
      FP (Faux Positif)  : alarme déclenchée sans bolus manqué
      FN (Faux Négatif)  : bolus manqué non détecté

    Args:
        alarm_normal        : alarmes sur la simulation SANS panne
        alarm_fault         : alarmes sur la simulation AVEC panne
        missed_bolus_labels : tableau bool des bolus réellement manqués
        bolus_steps_sorted  : liste des pas-bolus triés
        sample_time         : durée d'un pas (minutes)

    Returns:
        vp, fp, fn, precision, rappel, f1, delay, detected
    """
    n  = min(len(alarm_normal), len(alarm_fault), len(missed_bolus_labels))
    vp = fp = fn = 0
    delays = []

    # On analyse chaque intervalle entre deux bolus consécutifs
    for i in range(len(bolus_steps_sorted) - 1):
        bs = bolus_steps_sorted[i]      # Pas de début de la fenêtre (bolus i)
        be = bolus_steps_sorted[i + 1]  # Pas de fin (bolus suivant)

        if bs >= n:
            continue
        be = min(be, n)

        # Y a-t-il eu une alarme dans cette fenêtre ?
        normal_alerted = bool(np.any(alarm_normal[bs:be]))
        fault_alerted  = bool(np.any(alarm_fault[bs:be]))

        # Y a-t-il eu un bolus manqué à ce pas ?
        missed = bool(missed_bolus_labels[bs]) if bs < len(missed_bolus_labels) else False

        # Alarme sur la simulation normale → c'est un FP (fausse alarme)
        if normal_alerted:
            fp += 1

        if missed:
            if fault_alerted:
                vp += 1  # Bolus manqué et détecté → VP
                # Calcul du délai de détection
                idx = np.where(alarm_fault[bs:be])[0]
                if len(idx) > 0:
                    delays.append(idx[0] * sample_time)
            else:
                fn += 1  # Bolus manqué mais non détecté → FN
        else:
            if fault_alerted:
                fp += 1  # Pas de bolus manqué mais alarme quand même → FP

    # Calcul des métriques classiques
    precision = vp / (vp + fp) if (vp + fp) else 0.0
    rappel    = vp / (vp + fn) if (vp + fn) else 0.0
    f1        = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0.0
    delay     = float(np.mean(delays)) if delays else float("inf")
    detected  = len(delays) > 0

    return vp, fp, fn, precision, rappel, f1, delay, detected


# =============================================================================
# DÉLAI DE PREMIÈRE DÉTECTION PAR FENÊTRE
# =============================================================================
def compute_first_delay(alarm_fault, windows, sample_time):
    """
    Calcule le délai moyen de première détection sur toutes les fenêtres.

    Args:
        alarm_fault : tableau bool des alarmes sur la simulation avec panne
        windows     : liste de tuples (ws, we) des fenêtres de détection
        sample_time : durée d'un pas (minutes)

    Returns:
        (délai_moyen, détecté) : float et bool
    """
    delays = []
    for ws, we in windows:
        # Premier pas où l'alarme est active dans cette fenêtre
        idx = np.where(alarm_fault[ws:we + 1])[0]
        if len(idx) > 0:
            delays.append(idx[0] * sample_time)  # Délai en minutes

    return (float(np.mean(delays)) if delays else float("inf"), len(delays) > 0)


# =============================================================================
# ÉVALUATION D'UNE COMBINAISON DE PARAMÈTRES
# =============================================================================
def evaluate(scores_normal, scores_fault, cgm_normal, cgm_fault, missed_bolus,
             bolus_steps, sample_time, window_size, percentile, confirm_steps, slope_min):
    """
    Évalue une combinaison spécifique de paramètres et retourne ses métriques.

    Étapes :
      1. Calcule le seuil d'alarme à partir du percentile sur les scores normaux
      2. Applique le masque de pente pour filtrer les fausses montées
      3. Applique la confirmation sur les alarmes brutes
      4. Calcule les métriques de détection
      5. Calcule la fonction objectif (score à minimiser)

    Args:
        scores_normal  : scores Kalman sur simulation normale (référence)
        scores_fault   : scores Kalman sur simulation avec pannes
        cgm_normal     : glycémies de la simulation normale
        cgm_fault      : glycémies de la simulation avec pannes
        missed_bolus   : tableau bool des bolus manqués
        bolus_steps    : liste des pas-bolus triés
        sample_time    : durée d'un pas (minutes)
        window_size    : taille de la fenêtre Kalman
        percentile     : percentile pour le seuil
        confirm_steps  : nombre de pas de confirmation requis
        slope_min      : pente minimale pour valider une alarme

    Returns:
        dict avec toutes les métriques et paramètres de cette combinaison
    """
    # Seuil d'alarme calculé sur les scores de la simulation SANS panne
    # → si le score dépasse ce seuil en situation de panne, c'est une détection
    threshold = np.percentile(scores_normal, percentile)

    # Alarmes brutes : le score dépasse le seuil ?
    raw_normal = scores_normal > threshold
    raw_fault  = scores_fault  > threshold

    # On filtre par la pente — on n'alarme que si la glycémie monte vraiment
    raw_normal = raw_normal & compute_slope_mask(cgm_normal, sample_time, slope_min)
    raw_fault  = raw_fault  & compute_slope_mask(cgm_fault,  sample_time, slope_min)

    # On applique la confirmation — pas d'alarme sur un seul pas isolé
    alarm_normal = apply_confirmation(raw_normal, confirm_steps)
    alarm_fault  = apply_confirmation(raw_fault,  confirm_steps)

    # Calcul des métriques
    vp, fp, fn, prec, rec, f1, delay, detected = compute_metrics(
        alarm_normal, alarm_fault, missed_bolus, bolus_steps, sample_time
    )

    # Fonction objectif : on minimise ce score pour choisir les meilleurs paramètres
    # Un FN (raté) est 3× plus grave qu'un FP (fausse alarme)
    # Le délai est pénalisé légèrement (W_DELAY par minute)
    objective = W_FN * fn + W_FP * fp + (W_DELAY * delay if detected and np.isfinite(delay) else 9999)

    return {
        "window":    window_size,
        "percentile": percentile,
        "threshold": threshold,
        "confirm":   confirm_steps,
        "slope_min": slope_min,
        "VP": vp, "FP": fp, "FN": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "delay":     delay,
        "detected":  detected,
        "objective": objective,
        "nb_windows": int(np.sum(missed_bolus)),
    }


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
def main():
    print("\n=== Calibration Kalman optimisée — ADOLESCENT — 40 h ===")

    # --- Étape 1 : Simulation sans panne (référence) ---
    cgm_normal, _, _, _, sample_time = run_simulation(with_fault=False)

    # --- Étape 2 : Simulation avec pannes (scénario dégradé) ---
    cgm_fault, _, missed_bolus, bolus_steps, _ = run_simulation(with_fault=True)

    print(f"Sample time   : {sample_time} min")
    print(f"Durée         : {MAX_STEPS * sample_time:.0f} min ({MAX_STEPS * sample_time / 60:.1f} h)")
    print(f"Bolus manqués : {int(np.sum(missed_bolus))}")

    # --- Étape 3 : Grid Search — on teste toutes les combinaisons de paramètres ---
    all_results = []

    for w in FENETRES:
        # Calcul des scores Kalman pour cette taille de fenêtre
        scores_normal = kalman_frozen_scores(cgm_normal, sample_time, w)
        scores_fault  = kalman_frozen_scores(cgm_fault,  sample_time, w)

        for p in PERCENTILES:
            for c in CONFIRM_STEPS_LIST:
                for smin in SLOPE_MIN_LIST:
                    # Évaluation de cette combinaison de paramètres
                    all_results.append(
                        evaluate(scores_normal, scores_fault, cgm_normal, cgm_fault,
                                 missed_bolus, bolus_steps, sample_time, w, p, c, smin)
                    )

    # --- Étape 4 : Sélection de la meilleure combinaison ---

    # On garde d'abord les résultats qui ont détecté ET dans le délai acceptable
    valid = [r for r in all_results if r["detected"] and r["delay"] <= DELAI_MAX_ACCEPTABLE]

    # Si aucun résultat ne respecte le délai, on élargit à tous ceux qui ont détecté
    pool = valid if valid else [r for r in all_results if r["detected"]]

    # On prend le meilleur selon la fonction objectif (puis FN, FP, délai en tiebreak)
    best = min(pool, key=lambda r: (r["objective"], r["FN"], r["FP"], r["delay"]))

    # --- Affichage des résultats ---
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