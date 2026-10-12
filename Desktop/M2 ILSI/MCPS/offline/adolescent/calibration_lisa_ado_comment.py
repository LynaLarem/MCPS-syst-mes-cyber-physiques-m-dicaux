# =============================================================================
# Calibration offline LISA — Patient ADOLESCENT
#
# LISA = algorithme de détection de bolus manqués basé sur 3 métriques :
#   GFM   (Glucose Flux Metric)     : mesure la montée cumulée du glucose
#   IFM   (Insulin Flux Metric)     : mesure le déficit d'insuline active
#   Pente (slope)                   : mesure la vitesse de montée du glucose lissé
#
# Architecture :
#   - Deux simulations : une normale (référence), une avec pannes (détection)
#   - Calibration par grid search : on teste toutes les combinaisons de paramètres
#   - Évaluation par fenêtres inter-bolus : chaque repas est une fenêtre
#   - Sélection : minimise les fausses alarmes, maximise le rappel, minimise le délai
# =============================================================================

import numpy as np
from datetime import datetime
from collections import namedtuple

# Imports du simulateur de glycémie (simglucose)
from simglucose.simulation.env import T1DSimEnv           # Environnement de simulation T1D
from simglucose.simulation.scenario import Scenario        # Classe de base des scénarios
from simglucose.simulation.scenario import Action as MealAction  # Action repas (glucides)
from simglucose.patient.t1dpatient import T1DPatient       # Modèle patient diabétique
from simglucose.sensor.cgm import CGMSensor                # Capteur de glycémie continu
from simglucose.actuator.pump import InsulinPump           # Pompe à insuline


# =============================================================================
# IDENTITÉ DU PATIENT
# =============================================================================
PATIENT_NAME = "adolescent#001"  # Nom du patient dans la base de données simglucose

# =============================================================================
# PLANNING DES REPAS SUR UNE JOURNÉE
# 5 repas représentatifs d'un adolescent
# min_in_day : minute dans la journée simulée (600 min = 10h)
# carbs      : glucides ingérés (grammes)
# =============================================================================
REPAS_JOURNEE = [
    {"min_in_day":  60, "carbs": 45, "label": "Petit-dej 45g"},
    {"min_in_day": 180, "carbs": 60, "label": "Dejeuner  60g"},
    {"min_in_day": 300, "carbs": 20, "label": "Gouter    20g"},
    {"min_in_day": 420, "carbs": 55, "label": "Diner     55g"},
    {"min_in_day": 540, "carbs": 15, "label": "Collation 15g"},
]

# =============================================================================
# PARAMÈTRES INSULINE (spécifiques à l'adolescent)
# Un ado est moins sensible à l'insuline qu'un enfant → ICR et ISF plus élevés
# =============================================================================
ICR       = 15.0   # Insulin-to-Carb Ratio : 1 unité couvre 15g de glucides
ISF       = 60.0   # Insulin Sensitivity Factor : 1 unité baisse la glycémie de 60 mg/dL
CGM_CIBLE = 120.0  # Glycémie cible visée (mg/dL)
BOLUS_MIN = 0.3    # Dose bolus minimale autorisée (unités) — sécurité basse
BOLUS_MAX = 10.0   # Dose bolus maximale autorisée (unités) — sécurité haute

# =============================================================================
# PARAMÈTRES DE SIMULATION
# =============================================================================
BASAL            = 0.040  # Dose basale standard par pas (unités/min)
PREBOLUS_STEPS   = 8      # Le bolus est injecté 8 pas AVANT le repas (anticipation)
PANNE_DUREE      = 32     # Durée d'une panne de pompe (en pas)
MARCHE_DUREE     = 20     # Durée de bon fonctionnement entre deux pannes (en pas)
PANNE_START_STEP = 10     # Pas à partir duquel les pannes peuvent commencer
MINUTES_PAR_JOUR = 600    # Durée d'une journée simulée (600 min = 10 heures)
MAX_STEPS        = 800    # Nombre total de pas — 800 × 3 min ≈ 40 heures

# =============================================================================
# GRILLE DE CALIBRATION (grid search)
# On teste toutes les combinaisons de ces valeurs pour trouver les meilleurs
# paramètres de détection
# =============================================================================
LW_LIST = [10, 15, 20, 25, 30, 40, 50]  # Long Window  : fenêtre longue (tendance lente)
SW_LIST = [3, 4, 5, 6, 8]               # Short Window : fenêtre courte (tendance rapide)
# Règle : SW < LW toujours (sinon pas de sens de comparer court vs long)

P_GFM_LIST   = [70, 75, 80, 85, 88, 90, 92, 95]  # Percentiles testés pour le seuil GFM
P_IFM_LIST   = [30, 40, 50, 60, 70, 75, 80, 85]  # Percentiles testés pour le seuil IFM
P_SLOPE_LIST = [70, 75, 80, 85, 88, 90]           # Percentiles testés pour le seuil de pente

SCORE_MIN_LIST     = [2, 3]              # Score total minimum pour déclencher une alarme brute
CONFIRM_STEPS_LIST = [3, 5, 7, 10, 12]  # Nombre de pas consécutifs requis pour confirmer

# =============================================================================
# CRITÈRES DE QUALITÉ — seuils acceptables pour les fausses alarmes (FA)
# On cherche d'abord 0 FA, puis on accepte progressivement 1, 2, 3, 5...
# =============================================================================
FA_NIVEAUX = [
    (0,  "idéal      (FA = 0)"),
    (1,  "très bon   (FA ≤ 1)"),
    (2,  "bon        (FA ≤ 2)"),
    (3,  "correct    (FA ≤ 3)"),
    (5,  "acceptable (FA ≤ 5)"),
]
RAPPEL_MIN = 0.50  # On exige au minimum 50 % de détection des bolus manqués

# =============================================================================
# STRUCTURE DE DONNÉES
# Action regroupe les deux sorties de la pompe à chaque pas
# =============================================================================
Action = namedtuple("Action", ["basal", "bolus"])
# Exemple : Action(basal=0.040, bolus=3.0)


# =============================================================================
# GÉNÉRATION DES REPAS SUR TOUTE LA DURÉE DE LA SIMULATION
# =============================================================================
def build_repas(sample_time, max_steps):
    """
    Construit la liste complète de tous les repas sur la durée de la simulation.
    Les repas journaliers sont répétés sur autant de jours que nécessaire.

    Args:
        sample_time : durée d'un pas en minutes (ex : 3 min)
        max_steps   : nombre total de pas de simulation

    Returns:
        repas : liste de dicts {"min": ..., "carbs": ..., "label": ...}
    """
    repas     = []
    total_min = max_steps * sample_time              # Durée totale en minutes
    nb_jours  = int(total_min / MINUTES_PAR_JOUR) + 2  # +2 pour ne pas couper le dernier jour

    for jour in range(nb_jours):
        base_min = jour * MINUTES_PAR_JOUR  # Minute de départ du jour courant

        for r in REPAS_JOURNEE:
            # Minute absolue du repas dans la simulation
            repas.append({
                "min":   base_min + r["min_in_day"],
                "carbs": r["carbs"],
                "label": r["label"],
            })

    return repas


# =============================================================================
# CALCUL DU BOLUS POUR UN REPAS
# =============================================================================
def calculer_bolus(carbs, cgm_val):
    """
    Calcule la dose de bolus insuline à injecter pour couvrir un repas.

    Logique :
      1. Hypoglycémie (<70) → 0 (trop dangereux d'injecter)
      2. Bolus repas   = carbs / ICR
      3. Bolus correction = (glycémie - cible) / ISF  (seulement si glycémie ≥ 80)
      4. Résultat borné entre BOLUS_MIN et BOLUS_MAX

    Args:
        carbs   : glucides du repas (grammes)
        cgm_val : glycémie actuelle (mg/dL)

    Returns:
        dose bolus arrondie à 2 décimales (unités)
    """
    if cgm_val < 70.0:
        return 0.0  # Sécurité absolue : pas d'insuline en hypoglycémie

    bolus_repas      = carbs / ICR  # Dose pour couvrir les glucides
    bolus_correction = (cgm_val - CGM_CIBLE) / ISF if cgm_val >= 80.0 else 0.0  # Correction glycémique

    return round(max(BOLUS_MIN, min(BOLUS_MAX, bolus_repas + bolus_correction)), 2)


# =============================================================================
# SCÉNARIO PATIENT — gère l'apport en glucides dans le simulateur
# =============================================================================
class PatientScenario(Scenario):
    """
    Scénario de simulation qui reproduit le comportement alimentaire du patient.
    Le simulateur appelle get_action() à chaque pas pour savoir si le patient mange.
    """

    def __init__(self, start_time, repas, sample_time):
        """
        Args:
            start_time  : datetime de début de simulation
            repas       : liste de repas avec minute absolue et glucides
            sample_time : durée d'un pas en minutes
        """
        self.start_time  = start_time
        self.repas       = repas
        self.sample_time = sample_time
        # Dictionnaire pas → glucides pour accès instantané
        self.meal_carbs  = {int(r["min"] / sample_time): r["carbs"] for r in repas}

    def get_action(self, t):
        """
        Retourne les glucides ingérés au pas de temps t (0 si pas de repas).

        Args:
            t : datetime du pas courant

        Returns:
            MealAction avec les grammes de glucides à ce pas
        """
        elapsed = (t - self.start_time).total_seconds() / 60  # Temps écoulé en minutes
        step    = int(round(elapsed / self.sample_time))       # Numéro du pas
        return MealAction(meal=self.meal_carbs.get(step, 0))   # 0 si pas de repas

    def reset(self):
        pass  # Rien à réinitialiser


# =============================================================================
# DÉTECTION DE PANNE DE POMPE
# La pompe alterne cycliquement : panne → marche → panne → ...
# =============================================================================
def pump_is_failed(step):
    """
    Indique si la pompe est en panne au pas 'step'.

    Avant PANNE_START_STEP → toujours OK.
    Après → cycle : [PANNE_DUREE pas en panne] + [MARCHE_DUREE pas OK] en boucle.

    Args:
        step : numéro du pas courant

    Returns:
        True si la pompe est en panne, False sinon
    """
    if step < PANNE_START_STEP:
        return False  # Pas encore de panne

    position = (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE)
    return position < PANNE_DUREE  # En panne si dans la 1ère partie du cycle


# =============================================================================
# CONTRÔLEUR INSULINE BASALE (règles simples adaptatives)
# =============================================================================
def insulin_controller(cgm_val, bolus_plan):
    """
    Calcule la dose basale ajustée selon la glycémie actuelle.
    Augmente si hyperglycémie, réduit si hypoglycémie, coupe à 0 si <70.

    Args:
        cgm_val    : glycémie actuelle (mg/dL)
        bolus_plan : bolus prévu (retransmis sans modification)

    Returns:
        (basal_ajusté, bolus_plan)
    """
    db = BASAL  # Point de départ : dose basale standard

    # Ajustements progressifs selon la glycémie
    if cgm_val > 170: db += 0.007   # Hyperglycémie légère → légère augmentation
    if cgm_val > 210: db += 0.012   # Hyperglycémie sévère → augmentation supplémentaire
    if cgm_val < 100: db -= 0.007   # Tendance basse → réduction légère
    if cgm_val <  85: db -= 0.012   # Hypoglycémie proche → réduction forte
    if cgm_val <  70: db  = 0.0     # Hypoglycémie franche → coupure totale

    return max(0.0, db), bolus_plan  # On s'assure de ne jamais retourner un négatif


# =============================================================================
# SIMULATION COMPLÈTE
# =============================================================================
def run_simulation(with_fault=False):
    """
    Lance la simulation sur MAX_STEPS pas, avec ou sans pannes de pompe.

    Args:
        with_fault : si True → les pannes de pompe sont activées

    Returns:
        cgm_values       : tableau des glycémies à chaque pas (mg/dL)
        insulin_values   : tableau des doses d'insuline totales à chaque pas
        pump_failed_arr  : tableau bool — True si pompe en panne au pas i
        bolus_missed_arr : tableau bool — True si bolus raté au pas i (panne + repas)
        bolus_steps_list : liste triée des pas où un bolus était prévu
        sample_time      : durée d'un pas en minutes (lue depuis le simulateur)
    """
    start_time = datetime.now()

    # --- Récupération du sample_time réel du simulateur ---
    # On instancie un environnement temporaire juste pour lire cette valeur
    env_tmp = T1DSimEnv(
        patient  = T1DPatient.withName(PATIENT_NAME),
        sensor   = CGMSensor.withName("Dexcom", seed=1),
        pump     = InsulinPump.withName("Insulet"),
        scenario = PatientScenario(start_time, [{"min": 60, "carbs": 45}], 3)
    )
    sample_time = env_tmp.sample_time  # Ex : 3 minutes par pas
    del env_tmp  # L'environnement temporaire a rempli son rôle, on le supprime

    # --- Création de l'environnement réel ---
    repas = build_repas(sample_time, MAX_STEPS)  # Tous les repas sur 40h

    env = T1DSimEnv(
        patient  = T1DPatient.withName(PATIENT_NAME),
        sensor   = CGMSensor.withName("Dexcom", seed=1),
        pump     = InsulinPump.withName("Insulet"),
        scenario = PatientScenario(start_time, repas, sample_time)
    )

    # --- Calcul des pas-bolus (anticipés avant chaque repas) ---
    bolus_steps_map = {}
    for r in repas:
        step_repas = int(r["min"] / sample_time)          # Pas du repas
        step_bolus = max(0, step_repas - PREBOLUS_STEPS)  # Pas du bolus (avant le repas)
        bolus_steps_map[step_bolus] = r["carbs"]          # Glucides associés

    # --- Boucle principale de simulation ---
    s   = env.reset()     # Initialisation, retourne le premier état
    obs = s.observation   # Observation initiale (contient CGM, etc.)

    cgm_values       = []  # Historique des glycémies
    insulin_values   = []  # Historique des doses d'insuline totales
    pump_failed_arr  = []  # Historique état pompe (True = en panne)
    bolus_missed_arr = []  # Historique bolus manqués (True = bolus raté)
    bolus_steps_list = []  # Pas où un bolus était prévu

    for step in range(MAX_STEPS):
        cgm_val   = float(obs.CGM)  # Glycémie au pas courant

        carbs_now  = bolus_steps_map.get(step, 0)  # Glucides prévus à ce pas (0 si aucun)
        bolus_plan = calculer_bolus(carbs_now, cgm_val) if carbs_now > 0 else 0.0

        failed = pump_is_failed(step) if with_fault else False

        # On enregistre ce pas comme un "pas-bolus"
        if carbs_now > 0:
            bolus_steps_list.append(step)

        # Calcul des doses selon l'état de la pompe
        if failed:
            basal, bolus = 0.0, 0.0  # Pompe en panne → aucune injection
        else:
            basal, bolus = insulin_controller(cgm_val, bolus_plan)  # Injection normale

        # Insuline totale injectée à ce pas (basale sur la durée du pas + bolus)
        insulin_total = basal * sample_time + bolus

        # Enregistrement
        cgm_values.append(cgm_val)
        insulin_values.append(insulin_total)
        pump_failed_arr.append(failed)
        # Bolus raté = pompe HS exactement quand un bolus était attendu
        bolus_missed_arr.append(failed and carbs_now > 0)

        action = Action(basal=basal, bolus=bolus)
        s      = env.step(action)  # Avancer d'un pas dans le simulateur
        obs    = s.observation
        if s.done:
            break  # Fin anticipée si le simulateur le signale

    return (
        np.array(cgm_values),
        np.array(insulin_values),
        np.array(pump_failed_arr),
        np.array(bolus_missed_arr),
        sorted(bolus_steps_list),  # Trié pour les comparaisons inter-bolus
        sample_time,
    )


# =============================================================================
# MÉTRIQUES LISA
# =============================================================================

def compute_pie(insulin_values, alpha=0.85):
    """
    Calcule le PIE (Plasma Insulin Estimate) = insuline active estimée dans le sang.

    C'est un filtre exponentiel décroissant : l'insuline injectée s'accumule
    puis disparaît progressivement (alpha = taux de persistance).
    alpha=0.85 → l'insuline perd 15% de son effet à chaque pas.

    Args:
        insulin_values : tableau des doses d'insuline injectées à chaque pas
        alpha          : coefficient de persistance (entre 0 et 1)

    Returns:
        pie : tableau de l'insuline active estimée à chaque pas
    """
    pie = np.zeros(len(insulin_values))
    for i in range(1, len(insulin_values)):
        # L'insuline active = ce qui reste du pas précédent × alpha + injection du pas courant
        pie[i] = alpha * pie[i - 1] + insulin_values[i]
    return pie


def moving_average(x, window):
    """
    Calcule la moyenne glissante d'un signal sur une fenêtre de taille 'window'.
    Pour les premiers points (avant d'avoir assez d'historique), on moyenne
    sur les données disponibles (fenêtre partielle).

    Args:
        x      : signal d'entrée (tableau numpy ou liste)
        window : taille de la fenêtre (en nombre de pas)

    Returns:
        values : tableau de la moyenne glissante à chaque pas
    """
    values = np.zeros(len(x))
    for i in range(len(x)):
        start      = max(0, i - window + 1)  # Début de la fenêtre (au minimum 0)
        values[i]  = np.mean(x[start:i + 1]) # Moyenne sur la fenêtre courante
    return values


def compute_lisa_metrics(cgm_values, insulin_values, sample_time, lw_steps, sw_steps):
    """
    Calcule les 3 métriques LISA utilisées pour la détection :
      GFM   : Glucose Flux Metric — montée cumulée du glucose court vs long terme
      IFM   : Insulin Flux Metric — déficit d'insuline active court vs long terme
      Slope : pente du glucose lissé sur la fenêtre courte

    Intuition :
      - GFM élevé → le glucose monte plus vite récemment qu'en moyenne longue
      - IFM élevé → il y a moins d'insuline active récemment qu'en moyenne longue
      - Slope élevé → la glycémie monte rapidement en ce moment

    Args:
        cgm_values    : tableau des glycémies (mg/dL)
        insulin_values: tableau des doses totales d'insuline injectées
        sample_time   : durée d'un pas (minutes)
        lw_steps      : taille de la fenêtre longue (LW) en pas
        sw_steps      : taille de la fenêtre courte (SW) en pas

    Returns:
        gfm, ifm, slope : trois tableaux numpy de même taille
    """
    n = len(cgm_values)

    # --- Calcul de l'insuline active estimée ---
    pie = compute_pie(insulin_values)

    # --- Moyennes glissantes courtes et longues ---
    glucose_lw = moving_average(cgm_values, lw_steps)  # Glucose moyen sur longue fenêtre
    glucose_sw = moving_average(cgm_values, sw_steps)  # Glucose moyen sur courte fenêtre
    pie_lw     = moving_average(pie, lw_steps)          # Insuline active moyenne longue
    pie_sw     = moving_average(pie, sw_steps)          # Insuline active moyenne courte

    # --- GFM : Glucose Flux Metric ---
    # Mesure combien le glucose court terme dépasse le long terme, de façon cumulée
    # Si glucose_sw > glucose_lw → la glycémie monte → on accumule l'excès
    # Si glucose_sw ≤ glucose_lw → la glycémie est stable ou descend → on remet à 0
    gfm = np.zeros(n)
    for k in range(1, n):
        diff   = glucose_sw[k] - glucose_lw[k]  # Écart court - long
        gfm[k] = gfm[k - 1] + diff * sample_time if diff > 0 else 0.0
        # × sample_time pour convertir l'écart en une "aire" (intégrale discrète)

    # --- IFM : Insulin Flux Metric ---
    # Mesure le déficit d'insuline active : est-ce qu'il y a moins d'insuline
    # récemment (court terme) qu'en moyenne longue ?
    # Si pie_sw < pie_lw → déficit → IFM positif
    ifm = np.zeros(n)
    for k in range(n):
        if pie_lw[k] > 1e-6:  # On évite la division par zéro
            ifm[k] = max(0.0, (pie_lw[k] - pie_sw[k]) / pie_lw[k])
            # Normalisé entre 0 et 1 : 0 = pas de déficit, 1 = déficit total

    # --- Pente du glucose lissé (fenêtre courte) ---
    # Variation de la moyenne courte d'un pas à l'autre, divisée par le temps
    slope    = np.zeros(n)
    slope[1:] = np.diff(glucose_sw) / sample_time  # mg/dL par minute

    return gfm, ifm, slope


# =============================================================================
# CALCUL DES ALARMES LISA (score pondéré + confirmation)
# =============================================================================
def compute_alarms(gfm, ifm, slope,
                   seuil_gfm, seuil_ifm, seuil_slope,
                   score_min, confirm_steps):
    """
    Génère les alarmes LISA à partir des 3 métriques, d'un score pondéré
    et d'un filtre de confirmation.

    Système de score pondéré (max 4 points) :
      +2 si GFM   > seuil_gfm    (GFM est le plus important → poids double)
      +1 si IFM   > seuil_ifm
      +1 si Slope > seuil_slope
      total ≥ score_min → alarme brute

    Confirmation : l'alarme brute doit être active pendant au moins
    confirm_steps pas CONSÉCUTIFS pour déclencher une alarme confirmée.
    Cela évite les fausses alarmes causées par des pics isolés.

    Args:
        gfm, ifm, slope           : métriques LISA (tableaux numpy)
        seuil_gfm, seuil_ifm, seuil_slope : seuils de déclenchement
        score_min                 : score minimum pour alarme brute (2 ou 3)
        confirm_steps             : pas consécutifs requis pour confirmation

    Returns:
        confirmed : tableau bool — True si alarme confirmée au pas i
    """
    # Calcul du score à chaque pas (combinaison pondérée des 3 métriques)
    score = (2 * (gfm   > seuil_gfm  ).astype(int) +
             1 * (ifm   > seuil_ifm  ).astype(int) +
             1 * (slope > seuil_slope).astype(int))

    # Alarme brute si le score atteint le minimum requis
    raw_alarm = score >= score_min

    # Filtre de confirmation : on compte les pas consécutifs avec alarme brute
    confirmed = np.zeros(len(gfm), dtype=bool)
    consec    = 0
    for k in range(len(gfm)):
        consec       = consec + 1 if raw_alarm[k] else 0  # Incrément ou remise à 0
        confirmed[k] = consec >= confirm_steps             # Confirmée si assez de pas consécutifs

    return confirmed


# =============================================================================
# ÉVALUATION PAR FENÊTRES INTER-BOLUS
# =============================================================================
def evaluate_windows(alarms, bolus_missed_arr, bolus_steps_sorted):
    """
    Évalue les performances de détection fenêtre par fenêtre.
    Chaque fenêtre = intervalle [bolus_i → bolus_{i+1}[

    Règles strictes :
      - Bolus manqué + alarme dans la fenêtre → VP (Vrai Positif, bonne détection)
      - Bolus manqué + pas d'alarme           → FN (Faux Négatif, bolus raté non détecté)
      - Bolus présent + alarme dans la fenêtre → FP (Faux Positif, fausse alarme)
      - Dernière fenêtre sans bolus suivant   → ignorée

    Args:
        alarms              : tableau bool des alarmes au pas i
        bolus_missed_arr    : tableau bool — True si bolus manqué au pas i
        bolus_steps_sorted  : liste triée des pas où un bolus était prévu

    Returns:
        vp, fp, fn : compteurs de détection
    """
    vp = fp = fn = 0
    n  = len(alarms)

    for i in range(len(bolus_steps_sorted) - 1):
        bs = bolus_steps_sorted[i]      # Début de la fenêtre (bolus i)
        be = bolus_steps_sorted[i + 1]  # Fin de la fenêtre (bolus suivant)

        if bs >= n:
            continue
        be = min(be, n)  # On ne dépasse pas la fin des données

        # Y a-t-il eu une alarme quelque part dans cette fenêtre ?
        alerted = bool(np.any(alarms[bs:be]))
        # Le bolus de ce pas a-t-il été manqué ?
        missed  = bool(bolus_missed_arr[bs]) if bs < len(bolus_missed_arr) else False

        if missed:
            if alerted:
                vp += 1  # Bolus manqué ET détecté → bonne détection
            else:
                fn += 1  # Bolus manqué mais NON détecté → manqué
        else:
            if alerted:
                fp += 1  # Pas de bolus manqué mais alarme quand même → fausse alarme


    return vp, fp, fn


# =============================================================================
# ÉVALUATION D'UNE CONFIGURATION COMPLÈTE DE PARAMÈTRES
# =============================================================================
def evaluate_config(
    cgm_normal, insulin_normal, bolus_missed_normal, bolus_steps_sorted,
    cgm_fault,  insulin_fault,  bolus_missed_fault,
    sample_time, lw_steps, sw_steps,
    p_gfm, p_ifm, p_slope,
    score_min, confirm_steps
):
    """
    Évalue une combinaison précise de paramètres LISA et retourne ses métriques.

    Étapes :
      1. Calcule GFM, IFM, Slope pour les deux simulations (normale et panne)
      2. Calcule les seuils d'alarme à partir des percentiles sur les données normales
         → le seuil est calibré sur ce qu'on voit en situation normale (pas de panne)
      3. Génère les alarmes confirmées pour les deux simulations
      4. Évalue VP/FP/FN par fenêtres inter-bolus
         → FP = fausses alarmes sur le scénario normal + fausses alarmes sur panne
      5. Calcule rappel, précision, F1, délai de détection

    Args:
        cgm_normal, insulin_normal, bolus_missed_normal : données simulation normale
        cgm_fault,  insulin_fault,  bolus_missed_fault  : données simulation avec pannes
        bolus_steps_sorted : pas-bolus triés (communs aux deux simulations)
        sample_time        : durée d'un pas (minutes)
        lw_steps, sw_steps : tailles des fenêtres longue et courte
        p_gfm, p_ifm, p_slope : percentiles pour les seuils (ex : 85 → 85ème percentile)
        score_min          : score minimum pour alarme brute
        confirm_steps      : pas consécutifs requis pour confirmation

    Returns:
        dict complet avec paramètres, seuils, métriques et délai
    """
    # --- Calcul des métriques LISA sur les deux simulations ---
    gfm_n, ifm_n, slope_n = compute_lisa_metrics(
        cgm_normal, insulin_normal, sample_time, lw_steps, sw_steps)
    gfm_f, ifm_f, slope_f = compute_lisa_metrics(
        cgm_fault, insulin_fault, sample_time, lw_steps, sw_steps)

    # --- Calcul des seuils calibrés sur le scénario NORMAL ---
    # L'idée : on apprend ce qu'est "normal" sur la simulation sans panne,
    # puis on détecte les déviations dans la simulation avec panne.
    pos_slope_n = slope_n[slope_n > 0]  # On ne prend que les pentes positives pour le percentile
    seuil_gfm   = np.percentile(gfm_n, p_gfm)    # Seuil GFM : valeur dépassée p_gfm % du temps en situation normale
    seuil_ifm   = np.percentile(ifm_n, p_ifm)    # Seuil IFM : idem
    seuil_slope = np.percentile(pos_slope_n, p_slope) if len(pos_slope_n) > 0 else 0.0  # Seuil pente

    # --- Génération des alarmes confirmées ---
    alarms_normal = compute_alarms(
        gfm_n, ifm_n, slope_n, seuil_gfm, seuil_ifm, seuil_slope, score_min, confirm_steps)
    alarms_fault  = compute_alarms(
        gfm_f, ifm_f, slope_f, seuil_gfm, seuil_ifm, seuil_slope, score_min, confirm_steps)

    # --- Évaluation VP/FP/FN par fenêtres ---
    # FP total = fausses alarmes sur le scénario normal + sur les fenêtres non manquées du scénario panne
    _, fp_normal, _ = evaluate_windows(alarms_normal, bolus_missed_normal, bolus_steps_sorted)
    vp, fp_fault, fn = evaluate_windows(alarms_fault, bolus_missed_fault, bolus_steps_sorted)
    fp = fp_normal + fp_fault  # Total des fausses alarmes (les deux sources)

    # --- Métriques classiques ---
    rappel    = vp / max(1, vp + fn)                                   # Taux de détection
    precision = vp / max(1, vp + fp)                                   # Taux de précision
    f1        = 2 * precision * rappel / max(1e-6, precision + rappel) # Score F1

    # --- Délai de détection ---
    # On cherche le délai dans la PREMIÈRE fenêtre contenant un bolus manqué
    delay_min = float("inf")  # Infini si aucune détection
    detected  = False
    n_steps   = len(alarms_fault)

    for i, bs in enumerate(bolus_steps_sorted):
        be = bolus_steps_sorted[i + 1] if i + 1 < len(bolus_steps_sorted) else n_steps
        bs = min(bs, n_steps - 1)
        be = min(be, n_steps)

        # Ce bolus a-t-il été manqué ?
        if bs < len(bolus_missed_fault) and bolus_missed_fault[bs]:
            # On cherche le premier pas d'alarme dans cette fenêtre
            alarm_idxs = np.where(alarms_fault[bs:be])[0]
            if len(alarm_idxs) > 0:
                delay_min = alarm_idxs[0] * sample_time  # Délai en minutes depuis le bolus manqué
                detected  = True
                break  # On s'arrête à la première fenêtre détectée

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


# =============================================================================
# SÉLECTION DU MEILLEUR RÉSULTAT
# =============================================================================
def select_best(all_results):
    """
    Choisit la meilleure configuration parmi toutes celles testées.

    Stratégie de sélection en cascade :
      1. On cherche d'abord les configs qui ont détecté ET rappel ≥ 50%
         avec 0 fausse alarme (FA)
      2. Si aucune → on accepte jusqu'à 1 FA, puis 2, puis 3, puis 5
      3. Si toujours rien → on prend la meilleure par rappel (fallback)

    Parmi les candidats d'un niveau, on trie par :
      (FP croissant, rappel décroissant, délai croissant, confirm_steps décroissant)
      → on préfère peu de fausses alarmes, beaucoup de détections, vite, avec confirmation longue

    Args:
        all_results : liste de dicts résultats de evaluate_config

    Returns:
        (best, qualite_label) : meilleur résultat et label de qualité associé
    """
    for fa_max, fa_label in FA_NIVEAUX:
        # Filtrage des candidats respectant les critères du niveau courant
        candidats = [
            r for r in all_results
            if r["panne_detectee"]          # A détecté au moins une panne
            and r["rappel"] >= RAPPEL_MIN   # Détecte au moins 50% des bolus manqués
            and r["fp"] <= fa_max           # Fausses alarmes dans la limite du niveau
        ]
        if candidats:
            # On prend le meilleur candidat selon le tri multi-critères
            best = min(candidats, key=lambda x: (
                x["fp"],               # Moins de fausses alarmes
                -x["rappel"],          # Meilleur rappel (- car on minimise)
                x["delai_detection_min"],  # Délai le plus court
                -x["confirm_steps"],   # Confirmation la plus longue (plus robuste)
            ))
            return best, fa_label

    # Fallback : aucun critère strict satisfait → on prend juste le meilleur rappel
    candidats = [r for r in all_results if r["panne_detectee"]]
    if candidats:
        best = min(candidats, key=lambda x: (-x["rappel"], x["fp"]))
        return best, f"fallback — rappel={best['rappel']*100:.0f}%, FA={best['fp']}"

    return None, "aucune détection"  # Cas extrême : aucune config n'a détecté


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
def main():
    print("\n=== Calibration offline LISA — Patient ADOLESCENT — v4 ===")
    print("Évaluation : fenêtres inter-bolus (VP/FP/FN)")
    print("Logique    : score pondéré GFM×2 + IFM×1 + Pente×1 + confirmation")
    print("Repas      : multi-jours (5 repas/jour), bolus dynamique ICR/ISF")
    print()
    print("Simulation en cours...")

    # --- Étape 1 : Simulation normale (référence sans panne) ---
    (cgm_n, ins_n, _, bmissed_n, bolus_steps_sorted, sample_time
     ) = run_simulation(with_fault=False)

    # --- Étape 2 : Simulation avec pannes (scénario dégradé) ---
    (cgm_f, ins_f, _, bmissed_f, _, _
     ) = run_simulation(with_fault=True)

    print(f"Sample time   : {sample_time} min")
    print(f"Durée         : {len(cgm_n) * sample_time:.0f} min ({len(cgm_n) * sample_time / 60:.1f} h)")
    print(f"Bolus steps   : {len(bolus_steps_sorted)}")
    print(f"Bolus manqués : {int(np.sum(bmissed_f))} steps")
    print()

    # --- Étape 3 : Grid Search — test de toutes les combinaisons de paramètres ---
    all_results    = []
    total_configs  = (len(LW_LIST) * len(SW_LIST) * len(P_GFM_LIST) *
                      len(P_IFM_LIST) * len(P_SLOPE_LIST) *
                      len(SCORE_MIN_LIST) * len(CONFIRM_STEPS_LIST))
    print(f"Calibration : jusqu'à {total_configs} configurations à tester...")
    print()

    for lw in LW_LIST:
        for sw in SW_LIST:
            if sw >= lw:
                continue  # On ne teste que les cas où SW < LW (règle fondamentale LISA)

            for p_gfm in P_GFM_LIST:
                for p_ifm in P_IFM_LIST:
                    for p_slope in P_SLOPE_LIST:
                        for score_min in SCORE_MIN_LIST:
                            for confirm_steps in CONFIRM_STEPS_LIST:
                                # Évaluation de cette combinaison précise
                                r = evaluate_config(
                                    cgm_n, ins_n, bmissed_n, bolus_steps_sorted,
                                    cgm_f, ins_f, bmissed_f,
                                    sample_time, lw, sw,
                                    p_gfm, p_ifm, p_slope,
                                    score_min, confirm_steps
                                )
                                all_results.append(r)

    # --- Étape 4 : Sélection de la meilleure configuration ---
    best, qualite = select_best(all_results)

    print(f"Configurations testées : {len(all_results)}")
    print()
    print("=== Paramètres retenus ===")

    if best is None:
        print("Aucune configuration satisfaisante trouvée.")
        return

    # --- Affichage des résultats ---
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

    # --- Lignes directement copiables dans le code online ---
    print("─── Lignes à copier dans le code online ──────────────────────────────────")
    print(f"LISA_LW                 = {best['lw_steps']}")
    print(f"LISA_SW                 = {best['sw_steps']}")
    print(f"LISA_GFM_THRESHOLD      = {best['seuil_gfm']:.6f}")
    print(f"LISA_IFM_THRESHOLD      = {best['seuil_ifm']:.6f}")
    print(f"LISA_SLOPE_THRESHOLD    = {best['seuil_slope']:.6f}")
    print(f"LISA_SCORE_MIN          = {best['score_min']}")
    print(f"LISA_CONFIRMATION_STEPS = {best['confirm_steps']}")
    print()

    # --- Rappel de la logique de décision à implémenter en online ---
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