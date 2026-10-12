"""
sim_adulte_pannes.py
────────────────────────────────────────────
Simulation CGM continue — Adulte DT1
Pannes INTERMITTENTES : la pompe tombe en panne,
redémarre, retombe en panne, etc. tout au long de la simulation.

Paramètres des cycles (modifiables) :
  PANNE_DUREE   : nombre de steps où la pompe est HS
  MARCHE_DUREE  : nombre de steps où la pompe fonctionne

Visualisation :
  - Bandes rouges  = pompe HS
  - Courbe CGM colorée selon l'état de la pompe
  - Alarme clignotante lors des pannes
"""

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


# ═══════════════════════════════════════════════════════════
# REPAS ADULTE
# ═══════════════════════════════════════════════════════════

REPAS = [
    {"min":  60, "carbs": 50, "bolus": 6.0,  "label": "Petit-dej\n50g"},
    {"min": 180, "carbs": 70, "bolus": 8.5,  "label": "Dejeuner\n70g"},
    {"min": 300, "carbs": 25, "bolus": 3.0,  "label": "Gouter\n25g"},
    {"min": 420, "carbs": 60, "bolus": 7.5,  "label": "Diner\n60g"},
]

# ═══════════════════════════════════════════════════════════
# PANNES INTERMITTENTES
# Modifier ces deux valeurs pour ajuster les cycles
# ═══════════════════════════════════════════════════════════

PANNE_DUREE  = 30   # steps consécutifs avec pompe HS
MARCHE_DUREE = 25   # steps consécutifs avec pompe OK

# Premier step où les pannes commencent
# (on laisse quelques steps normaux au début)
PANNE_START_STEP = 10


def pump_is_failed(step):
    """Retourne True si la pompe est en panne à ce step."""
    if step < PANNE_START_STEP:
        return False
    cycle_pos = (step - PANNE_START_STEP) % (PANNE_DUREE + MARCHE_DUREE)
    return cycle_pos < PANNE_DUREE


# ═══════════════════════════════════════════════════════════
# SCENARIO
# ═══════════════════════════════════════════════════════════

class AdultScenario(Scenario):
    def __init__(self, start_time, repas, sample_time):
        self.start_time  = start_time
        self.repas       = repas
        self.sample_time = sample_time
        self.meal_steps  = {
            int(r["min"] / sample_time): r["carbs"]
            for r in repas
        }

    def get_action(self, t):
        elapsed = (t - self.start_time).total_seconds() / 60
        step    = int(round(elapsed / self.sample_time))
        carbs   = self.meal_steps.get(step, 0)
        return MealAction(meal=carbs)

    def reset(self):
        pass


Action = namedtuple("Action", ["basal", "bolus"])


# ═══════════════════════════════════════════════════════════
# INITIALISATION PATIENT
# ═══════════════════════════════════════════════════════════

patient    = T1DPatient.withName("adult#001")
sensor     = CGMSensor.withName("Dexcom", seed=1)
pump       = InsulinPump.withName("Insulet")
start_time = datetime.now()

env_tmp = T1DSimEnv(
    patient=patient, sensor=sensor, pump=pump,
    scenario=AdultScenario(start_time, REPAS, 3)
)
sample_time = env_tmp.sample_time
del env_tmp

patient  = T1DPatient.withName("adult#001")
sensor   = CGMSensor.withName("Dexcom", seed=1)
pump     = InsulinPump.withName("Insulet")
scenario = AdultScenario(start_time, REPAS, sample_time)

env = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

for r in REPAS:
    r["step"] = int(r["min"] / sample_time)

PREBOLUS_STEPS = 8
BOLUS_STEPS    = {
    max(0, r["step"] - PREBOLUS_STEPS): r["bolus"]
    for r in REPAS
}

BASAL      = 0.050
STEP_DELAY = 1.5

print(f"Patient      : adult#001")
print(f"Sample time  : {sample_time} min")
print(f"Basal        : {BASAL} U/min")
print(f"Cycle pannes : {PANNE_DUREE} steps HS → {MARCHE_DUREE} steps OK → ...")
print(f"Début pannes : step {PANNE_START_STEP} "
      f"({PANNE_START_STEP * sample_time:.0f} min)")
print("\nRepas :")
for r in REPAS:
    bs = max(0, r["step"] - PREBOLUS_STEPS)
    print(f"  step {r['step']:>4}  {r['label'].split(chr(10))[0]:<12}")
print("\nCtrl+C pour arrêter\n")


# ═══════════════════════════════════════════════════════════
# GRAPHE
# ═══════════════════════════════════════════════════════════

plt.style.use("default")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

times      = []
cgm_values = []
pump_states = []   # True = panne, False = OK

line_cgm,  = ax.plot([], [], color="#0057b7", linewidth=2.5, zorder=3)
fill_ref    = [None]

scatter_meals = ax.scatter(
    [], [], marker="^", s=220, color="#cc5500",
    zorder=5, edgecolors="white", linewidths=0.8
)
scatter_insulin_ok = ax.scatter(
    [], [], marker="o", s=160, color="#003f8a",
    zorder=5, edgecolors="white", linewidths=0.8,
    label="Bolus injecté (pompe OK)"
)
scatter_insulin_fail = ax.scatter(
    [], [], marker="x", s=200, color="#cc0000",
    zorder=5, linewidths=2.5,
    label="Bolus manqué (panne)"
)

ax.axhline(y=70,  color="#003f8a", linestyle="--", linewidth=1.2, alpha=0.7)
ax.axhline(y=180, color="#cc0000", linestyle="--", linewidth=1.2, alpha=0.7)
ax.axhspan(70, 180, alpha=0.08, color="#e8f4fd")

ax.set_ylim(40, 500)
ax.set_xlabel("Heure simulée", color="#333333")
ax.set_ylabel("Glycémie (mg/dL)", color="#333333")
ax.tick_params(colors="#333333")

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["bottom", "left"]:
    ax.spines[s].set_color("#aaaaaa")

ax.grid(True, color="#cccccc", alpha=0.6, linewidth=0.5)

fig.suptitle(
    "CGM Continue — Adulte DT1  |  Pannes pompe intermittentes",
    fontsize=12, color="#111111", fontfamily="monospace"
)

legend_elems = [
    Line2D([0], [0], color="#0057b7", lw=2.5, label="CGM"),
    Patch(facecolor="#cc0000", alpha=0.15, label=f"Panne pompe ({PANNE_DUREE} steps)"),
    Patch(facecolor="#e6f7ee", alpha=0.7, label=f"Pompe OK ({MARCHE_DUREE} steps)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#cc5500",
           markersize=10, label="Repas", linestyle="None"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#003f8a",
           markersize=9,  label="Bolus injecté", linestyle="None"),
    Line2D([0], [0], marker="x", color="#cc0000",
           markersize=10, label="Bolus manqué (panne)", linestyle="None"),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=8,
          facecolor="white", edgecolor="#aaaaaa")

text_val = ax.text(
    0.99, 0.97, "",
    transform=ax.transAxes, color="#0057b7", fontsize=15,
    ha="right", va="top", fontfamily="monospace", fontweight="bold"
)
text_info = ax.text(
    0.99, 0.90, "",
    transform=ax.transAxes, color="#444444", fontsize=9,
    ha="right", va="top"
)
text_alarm = ax.text(
    0.70, 0.60, "",
    transform=ax.transAxes, color="#cc0000", fontsize=11,
    ha="center", va="center", fontfamily="monospace", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe5e5",
              edgecolor="#cc0000", alpha=0.0)
)

# Bandes de fond (panne/marche) — recalculées à chaque step
failure_bands = []

plt.tight_layout()
plt.ion()
plt.show()


# ═══════════════════════════════════════════════════════════
# RESET
# ═══════════════════════════════════════════════════════════

s      = env.reset()
obs    = s.observation
reward = s.reward
done   = s.done
info   = s.info

print(f"{'Step':>5} {'Heure':>6} {'CGM':>8} "
      f"{'Repas':>8} {'Bolus':>8} {'Pompe':>10}")
print("─" * 68)

index      = 0
prev_state = None   # pour détecter les transitions


# ═══════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════════════

try:
    while True:

        cgm_val    = float(obs.CGM)
        sim_min    = index * sample_time
        time_label = (start_time + timedelta(minutes=sim_min)).strftime("%H:%M")

        carbs      = scenario.meal_steps.get(index, 0)
        bolus_plan = BOLUS_STEPS.get(index, 0)
        failed     = pump_is_failed(index)

        # ── Détection transitions (affichage console) ──────
        if prev_state is not None and failed != prev_state:
            if failed:
                print()
                print("⚠️  " * 12)
                print("  PANNE POMPE — Aucune insuline délivrée")
                print("⚠️  " * 12)
                print()
            else:
                print()
                print("✅  " * 12)
                print("  POMPE RÉTABLIE — Insuline à nouveau active")
                print("✅  " * 12)
                print()
        prev_state = failed

        # ── Zone glycémique ────────────────────────────────
        if cgm_val < 70:
            color = "#003f8a"; zone = "HYPO"
        elif cgm_val > 180:
            color = "#cc0000"; zone = "HYPER"
        else:
            color = "#0057b7"; zone = "NORMAL"

        pompe_tag = "PANNE ✗" if failed else "OK ✓"

        print(
            f"[{index:>4}] {time_label} "
            f"{cgm_val:>8.1f} "
            f"{str(carbs)+'g' if carbs else '—':>8} "
            f"{'—' if (not bolus_plan or failed) else str(bolus_plan)+'U':>8} "
            f"{pompe_tag:>10}"
        )

        times.append(time_label)
        cgm_values.append(cgm_val)
        pump_states.append(failed)

        MAX_PTS    = 80
        disp_times  = times[-MAX_PTS:]
        disp_cgm    = cgm_values[-MAX_PTS:]
        disp_states = pump_states[-MAX_PTS:]
        x           = list(range(len(disp_times)))
        offset      = max(0, len(times) - MAX_PTS)

        line_cgm.set_data(x, disp_cgm)

        if fill_ref[0] is not None:
            fill_ref[0].remove()
        fill_ref[0] = ax.fill_between(
            x, disp_cgm, alpha=0.10, color="#0057b7", zorder=2
        )

        ax.set_xlim(-1, len(disp_times))

        ticks = list(range(0, len(disp_times), 6))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [disp_times[i] for i in ticks if i < len(disp_times)],
            fontsize=8, color="#444444"
        )

        # ── Bandes panne/marche ────────────────────────────
        for band in failure_bands:
            band.remove()
        failure_bands.clear()

        # Calculer les segments consécutifs
        i = 0
        while i < len(disp_states):
            seg_state = disp_states[i]
            j = i
            while j < len(disp_states) and disp_states[j] == seg_state:
                j += 1
            color_band = "#fc8a8a" if seg_state else "#e6f7ee"
            alpha_band = 0.10 if seg_state else 0.04
            band = ax.axvspan(i - 0.5, j - 0.5,
                              alpha=alpha_band, color=color_band, zorder=1)
            failure_bands.append(band)
            i = j

        # ── Marqueurs repas ────────────────────────────────
        m_x, m_y = [], []
        for r in REPAS:
            vi = r["step"] - offset
            if 0 <= vi < len(disp_times):
                m_x.append(vi)
                m_y.append(disp_cgm[vi] + 18)
        scatter_meals.set_offsets(
            np.c_[m_x, m_y] if m_x else np.empty((0, 2))
        )

        # ── Marqueurs bolus OK / manqués ───────────────────
        i_ok_x, i_ok_y   = [], []
        i_fail_x, i_fail_y = [], []
        for step_bolus, dose in BOLUS_STEPS.items():
            vi = step_bolus - offset
            if 0 <= vi < len(disp_times):
                if pump_is_failed(step_bolus):
                    i_fail_x.append(vi)
                    i_fail_y.append(disp_cgm[vi] - 18)
                else:
                    i_ok_x.append(vi)
                    i_ok_y.append(disp_cgm[vi] - 18)
        scatter_insulin_ok.set_offsets(
            np.c_[i_ok_x, i_ok_y] if i_ok_x else np.empty((0, 2))
        )
        scatter_insulin_fail.set_offsets(
            np.c_[i_fail_x, i_fail_y] if i_fail_x else np.empty((0, 2))
        )

        # ── Textes ─────────────────────────────────────────
        text_val.set_text(f"{cgm_val:.0f} mg/dL [{zone}]")
        text_val.set_color(color)
        text_info.set_text(
            f"{time_label} | step {index} | "
            f"{'⚠ PANNE' if failed else '✓ Pompe OK'}"
        )

        if failed:
            text_alarm.set_text("⚠ PANNE POMPE\nAUCUNE INSULINE")
            text_alarm.get_bbox_patch().set_alpha(0.90)
        else:
            text_alarm.set_text("")
            text_alarm.get_bbox_patch().set_alpha(0.0)

        # ═══════════════════════════════════════════════════
        # CONTRÔLE INSULINE
        # ═══════════════════════════════════════════════════

        correction = 0.0
        if cgm_val > 160: correction += 0.4
        if cgm_val > 180: correction += 0.6
        if cgm_val > 200: correction += 1.0
        if cgm_val > 230: correction += 1.5
        if cgm_val < 110: correction = max(0.0, correction - 0.5)
        if cgm_val < 95:  correction = 0.0

        dynamic_basal = BASAL
        if cgm_val > 170: dynamic_basal += 0.008
        if cgm_val > 210: dynamic_basal += 0.015
        if cgm_val < 100: dynamic_basal -= 0.008
        if cgm_val < 85:  dynamic_basal -= 0.015
        dynamic_basal = max(0.01, dynamic_basal)

        total_bolus = bolus_plan + correction

        if failed:
            action = Action(basal=0.0, bolus=0.0)
        else:
            action = Action(basal=dynamic_basal, bolus=total_bolus)

        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass

        time.sleep(STEP_DELAY)

        if done:
            print("\n[INFO] Reset environnement\n")
            patient2  = T1DPatient.withName("adult#001")
            sensor2   = CGMSensor.withName("Dexcom", seed=index)
            pump2     = InsulinPump.withName("Insulet")
            scenario2 = AdultScenario(
                start_time + timedelta(minutes=index * sample_time),
                REPAS, sample_time
            )
            env2 = T1DSimEnv(patient=patient2, sensor=sensor2,
                             pump=pump2, scenario=scenario2)
            s   = env2.reset()
            env = env2
        else:
            s = env.step(action)

        obs    = s.observation
        reward = s.reward
        done   = s.done
        info   = s.info
        index += 1

except KeyboardInterrupt:
    print(f"\nSimulation arrêtée à step {index}")

plt.ioff()
plt.show()