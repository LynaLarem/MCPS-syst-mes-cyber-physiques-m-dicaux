# ============================================================
# sim_ado_simple.py
# Simulation CGM continue — Adolescent DT1 (adolescent#001)
# simglucose uniquement — graphe CGM seul
# ============================================================

import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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


# ============================================================
# CONFIGURATION ADOLESCENT
# ============================================================

REPAS = [
    {"min":  60, "carbs": 45, "bolus": 5.5,  "label": "Petit-dej\n45g"},
    {"min": 180, "carbs": 60, "bolus": 7.0,  "label": "Dejeuner\n60g"},
    {"min": 300, "carbs": 20, "bolus": 2.8,  "label": "Gouter\n20g"},
    {"min": 420, "carbs": 55, "bolus": 6.5,  "label": "Diner\n55g"},
]

BASAL          = 0.040   # U/min , valeur de base
PREBOLUS_STEPS = 8     # nombre de steps avant le repas (ex: 8 steps x 3 min = 24 min avant le repas)
STEP_DELAY     = 1.5   #une pause visuelle entre chaque step pour que le graphe temps réel soit lisible à l'écran
Action = namedtuple("Action", ["basal", "bolus"])

C_CGM    = "#7c3aed"
C_BOLUS  = "#d97706"
C_REPAS  = "#be185d"
C_HYPO   = "#1d4ed8"
C_HYPER  = "#b91c1c"
C_ZONE   = "#ede9fe"
C_TEXT   = "#1e293b"
C_GRID   = "#e2e8f0"
C_BORDER = "#cbd5e1"


# ============================================================
# SCENARIO
# AdoScenario hérite de la classe Scenario de SimGlucose.
# Son rôle est de dire à SimGlucose "à quel step il y a un repas et combien de glucides".
# SimGlucose appelle get_action() à chaque step pour savoir s'il y a un repas.

class AdoScenario(Scenario):
    def __init__(self, start_time, repas, sample_time):
        self.start_time  = start_time  # heure de début de la simulation
        self.repas       = repas     # liste des repas (min, carbs, label...)
        self.sample_time = sample_time    # durée d'un step en minutes (= 3 min)
        
        
        
        # Convertit la liste des repas en dictionnaire :
           # clé = numéro du step où le repas a lieu
           # valeur = quantité de glucides (g)
        # Exemple : {20: 45, 60: 60, 100: 20, 140: 55}
        # → au step 20 il y a 45g de glucides, au step 60 il y a 60g, etc.
        self.meal_steps  = {int(r["min"] / sample_time): r["carbs"] for r in repas}
        

    def get_action(self, t):
        # SimGlucose appelle cette méthode à chaque step en passant l'heure actuelle (t).
        # On calcule le numéro du step courant à partir de l'heure.
        elapsed = (t - self.start_time).total_seconds() / 60   # minutes écoulées depuis le début
        step    = int(round(elapsed / self.sample_time))        # numéro du step correspondant
        
         # Si ce step correspond à un repas → on retourne les glucides
        # Sinon → on retourne 0 (pas de repas ce step)
        return MealAction(meal=self.meal_steps.get(step, 0))

    def reset(self):
        pass


# ============================================================
# INITIALISATION
# ============================================================


# --- Création des composants du pancréas artificiel ---
patient    = T1DPatient.withName("adolescent#001")  # patient virtuel adolescent
sensor     = CGMSensor.withName("Dexcom", seed=2)   # capteur CGM Dexcom (seed=2 → bruit fixe reproductible)
pump       = InsulinPump.withName("Insulet")    # pompe à insuline Insulet
start_time = datetime.now()        # heure de début



# --- simulation temporaire juste pour récupérer sample_time ---
# On ne connaît pas encore sample_time (il dépend du capteur Dexcom dans SimGlucose).
# On crée un environnement temporaire avec un scénario minimal juste pour lire cette valeur.
# Puis on le supprime immédiatement avec del.
env_tmp    = T1DSimEnv(patient=patient, sensor=sensor, pump=pump,
                        scenario=AdoScenario(start_time, REPAS, 3)) # 3 = valeur provisoire
sample_time = env_tmp.sample_time    # → récupère la vraie valeur (3 min pour Dexcom)
del env_tmp   # supprime l'environnement temporaire


# --- Création du vrai environnement de simulation avec le bon sample_time ---
patient  = T1DPatient.withName("adolescent#001")
sensor   = CGMSensor.withName("Dexcom", seed=2)
pump     = InsulinPump.withName("Insulet")
scenario = AdoScenario(start_time, REPAS, sample_time)
env      = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)


# --- Calcul des steps de repas ---
# Pour chaque repas, on calcule à quel step il tombe.
# Exemple : repas à la minute 180, sample_time=3 → step 60
for r in REPAS:
    r["step"] = int(r["min"] / sample_time)
    
# --- Calcul des steps de bolus (PREBOLUS) ---
# Le bolus s'injecte PREBOLUS_STEPS = 8 steps avant le repas (= 24 min avant).
# BOLUS_STEPS = dictionnaire : clé = step d'injection, valeur = dose bolus
# max(0, ...) évite d'avoir un step négatif si le repas est très tôt.
# Exemple : repas au step 60 → bolus au step max(0, 60-8) = step 52
BOLUS_STEPS = {max(0, r["step"] - PREBOLUS_STEPS): r["bolus"] for r in REPAS}


# ============================================================
# GRAPHIQUE — CGM seul
# ============================================================

plt.style.use("default")
fig, ax_cgm = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor("white")

ax_cgm.set_facecolor("#faf5ff")
ax_cgm.grid(True, color=C_GRID, linewidth=0.8)
ax_cgm.tick_params(colors=C_TEXT, labelsize=9)
for sp in ["top", "right"]:
    ax_cgm.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax_cgm.spines[sp].set_color(C_BORDER)

line_cgm, = ax_cgm.plot([], [], color=C_CGM, lw=2.8, zorder=4)
fill_cgm  = [None]

ax_cgm.axhline(70,  color=C_HYPO,  ls="--", lw=1.2, alpha=0.7)
ax_cgm.axhline(180, color=C_HYPER, ls="--", lw=1.2, alpha=0.7)
ax_cgm.axhspan(70, 180, alpha=0.10, color=C_ZONE, zorder=1)

sc_meals = ax_cgm.scatter([], [], marker="^", s=200, color=C_REPAS,
                           zorder=6, edgecolors="white", linewidths=0.8)
sc_bolus = ax_cgm.scatter([], [], marker="o", s=140, color=C_BOLUS,
                           zorder=6, edgecolors="white", linewidths=0.8)

ax_cgm.set_ylim(50, 380)
ax_cgm.set_ylabel("Glycémie (mg/dL)", color=C_TEXT, fontsize=11, fontweight="bold")
ax_cgm.set_xlabel("Heure simulée", color=C_TEXT, fontsize=10)

legend_elems = [
    Line2D([0],[0], color=C_CGM,  lw=2.8, label="CGM simglucose"),
    Line2D([0],[0], color=C_HYPO, lw=1.2, ls="--", label="Hypoglycémie (70)"),
    Line2D([0],[0], color=C_HYPER,lw=1.2, ls="--", label="Hyperglycémie (180)"),
    Patch(facecolor=C_ZONE, alpha=0.6, label="Zone cible 70–180"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor=C_REPAS, ms=10, ls="None", label="Repas"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_BOLUS, ms=9,  ls="None", label="Bolus insuline"),
]
ax_cgm.legend(handles=legend_elems, loc="upper left", fontsize=9,
              facecolor="white", edgecolor=C_BORDER, framealpha=0.9)

txt_val  = ax_cgm.text(0.99, 0.97, "", transform=ax_cgm.transAxes,
                        color=C_CGM, fontsize=15, ha="right", va="top",
                        fontfamily="monospace", fontweight="bold")
txt_info = ax_cgm.text(0.99, 0.87, "", transform=ax_cgm.transAxes,
                        color="#64748b", fontsize=9, ha="right", va="top")

fig.suptitle("Simulation CGM — Adolescent DT1  (adolescent#001)",
             fontsize=13, color=C_TEXT, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.ion()
plt.show()


# ============================================================
# HISTORIQUES
# ============================================================

times      = []
cgm_vals   = []
_bolus_sum = 0.0
MAX_PTS    = 80


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

s     = env.reset()
obs   = s.observation
done  = s.done
index = 0

print(f"\n{'='*60}")
print(f"  Simulation Adolescent DT1 — adolescent#001")
print(f"  Sample time : {sample_time} min   Basal : {BASAL} U/min")
print(f"{'='*60}")
print(f"{'Step':>5} {'Heure':>6} {'CGM':>8} {'Zone':>7} {'Basal':>8} {'Bolus':>7} {'Repas':>7}")
print("─" * 60)

try:
    while True:
        cgm_val    = float(obs.CGM)
        sim_min    = index * sample_time
        time_label = (start_time + timedelta(minutes=sim_min)).strftime("%H:%M")

        carbs      = scenario.meal_steps.get(index, 0)
        bolus_plan = BOLUS_STEPS.get(index, 0)

        correction = 0.0
        if cgm_val > 155: correction += 0.4
        if cgm_val > 175: correction += 0.6
        if cgm_val > 200: correction += 1.0
        if cgm_val > 230: correction += 1.5
        if cgm_val < 115: correction = max(0.0, correction - 0.5)
        if cgm_val < 95:  correction = 0.0

        dynamic_basal = BASAL
        if cgm_val > 170: dynamic_basal += 0.007
        if cgm_val > 210: dynamic_basal += 0.012
        if cgm_val < 105: dynamic_basal -= 0.007
        if cgm_val < 85:  dynamic_basal -= 0.012
        dynamic_basal = max(0.010, dynamic_basal)

        total_bolus = bolus_plan + correction
        _bolus_sum += total_bolus

        zone = ("HYPO" if cgm_val < 70 else "HYPER" if cgm_val > 180 else "NORMAL")
        print(f"[{index:>4}] {time_label} {cgm_val:>8.1f} {zone:>7} "
              f"{dynamic_basal:>8.4f} {total_bolus:>7.2f} "
              f"{str(carbs)+'g' if carbs else '—':>7}")

        times.append(time_label)
        cgm_vals.append(cgm_val)

        disp_times = times[-MAX_PTS:]
        disp_cgm   = cgm_vals[-MAX_PTS:]
        x      = list(range(len(disp_times)))
        offset = max(0, len(times) - MAX_PTS)

        line_cgm.set_data(x, disp_cgm)

        if fill_cgm[0] is not None:
            fill_cgm[0].remove()
        fill_cgm[0] = ax_cgm.fill_between(x, disp_cgm, 50,
                                           alpha=0.07, color=C_CGM, zorder=2)

        if disp_cgm:
            ax_cgm.set_ylim(max(40, min(disp_cgm) - 30),
                            min(420, max(disp_cgm) + 60))

        ticks = list(range(0, len(disp_times), 6))
        ax_cgm.set_xlim(-1, len(disp_times))
        ax_cgm.set_xticks(ticks)
        ax_cgm.set_xticklabels([disp_times[i] for i in ticks if i < len(disp_times)],
                                 fontsize=8, color=C_TEXT)

        mx, my = [], []
        for r in REPAS:
            vi = r["step"] - offset
            if 0 <= vi < len(disp_times):
                mx.append(vi); my.append(disp_cgm[vi] + 18)
        sc_meals.set_offsets(np.c_[mx, my] if mx else np.empty((0, 2)))

        bx, by = [], []
        for step_b in BOLUS_STEPS:
            vi = step_b - offset
            if 0 <= vi < len(disp_times):
                bx.append(vi); by.append(disp_cgm[vi] - 18)
        sc_bolus.set_offsets(np.c_[bx, by] if bx else np.empty((0, 2)))

        col = C_HYPO if cgm_val < 70 else C_HYPER if cgm_val > 180 else C_CGM
        txt_val.set_text(f"{cgm_val:.0f} mg/dL  [{zone}]")
        txt_val.set_color(col)
        txt_info.set_text(f"{time_label} | step {index} | "
                          f"basal={dynamic_basal:.3f} U/min | "
                          f"bolus cumulé={_bolus_sum:.1f} U")

        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass

        time.sleep(STEP_DELAY)

        action = Action(basal=dynamic_basal, bolus=total_bolus)

        if done:
            print("\n[INFO] Reset\n")
            p2  = T1DPatient.withName("adolescent#001")
            s2  = CGMSensor.withName("Dexcom", seed=index)
            sc2 = AdoScenario(start_time + timedelta(minutes=index * sample_time),
                               REPAS, sample_time)
            env2 = T1DSimEnv(patient=p2, sensor=s2,
                              pump=InsulinPump.withName("Insulet"), scenario=sc2)
            s = env2.reset(); env = env2
        else:
            s = env.step(action)

        obs  = s.observation
        done = s.done
        index += 1

except KeyboardInterrupt:
    tir = sum(1 for v in cgm_vals if 70 <= v <= 180) / max(1, len(cgm_vals)) * 100
    print(f"\n{'='*60}")
    print(f"  Arrêt step {index} ({index * sample_time / 60:.1f} h simulées)")
    print(f"  TIR (70-180) : {tir:.1f}%")
    print(f"  Bolus total  : {_bolus_sum:.1f} U")
    print()

plt.ioff()
plt.show()