# Runtime Monitoring for Insulin Pump Fault Detection in an Artificial Pancreas

Fault detection for the insulin pump of an artificial pancreas (type 1 diabetes), based on runtime monitoring of continuous glucose monitor (CGM) data and insulin delivery data. The system is evaluated in simulation on three virtual patients (child, adolescent, adult).

This repository accompanies the Master's thesis *"Design and implementation of a fault-tolerance system based on runtime monitoring for a medical cyber-physical system"* (Université Constantine 2 – Abdelhamid Mehri, June 2026; thesis written in French).

## Why this problem

In an artificial pancreas, a pump failure (occlusion, loss of actuation, full or partial stop) leads to missed insulin and can cause severe hyperglycemia or ketoacidosis. The goal is to detect such failures early from the signals already available in the closed loop, without labelled data, and to give interpretable alerts.

## Approach

The system works in two phases.

**1. Offline calibration.** For each patient profile, window sizes and detection thresholds are calibrated by running normal and faulty scenarios in the simulator.

**2. Online detection.** Two detectors run in parallel on each new CGM sample and insulin dose:

- **Kalman-filter detector.** Compares measured and predicted glucose and analyses the residuals with a sliding window. For the adult profile, a CUSUM score accumulates positive residuals to catch slow drifts. A temporary freeze of the filter correction prevents abnormal CGM values from absorbing the fault.
- **LISA-based detector** (Loss in Infusion Set Actuation, Howsmon et al., 2017). Uses the Glucose Fault Metric (GFM), the Insulin Fault Metric (IFM) and the glucose slope over a short and a long window. We extended it with a weighted score (GFM counts 2 points, IFM and slope 1 point each) and a temporal confirmation counter.

A **decision-fusion module** combines both detectors into three confidence levels: normal operation, suspected failure and confirmed failure. Alerts are sent as phone notifications.

## Results (24-hour simulation, 6 failure windows)

| Method | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Kalman filter | 5 | 2 | 1 | 71.4 % | 83.3 % | 76.9 % |
| LISA (weighted score) | 5 | 2 | 1 | 71.4 % | 83.3 % | 76.9 % |
| Decision fusion | 4 | 1 | 2 | 80.0 % | 66.7 % | 72.7 % |

The two detectors catch different failures, which motivates the hybrid design. Fusion gives higher precision at the cost of recall. These results come from a single 24-hour simulation and should be read as a proof of concept.

## Repository structure

<!-- Adapt this section to your actual files -->

```
.
├── <detector_kalman>.py      # Kalman-filter detector (sliding window, CUSUM, freeze)
├── <detector_lisa>.py        # LISA-based detector (GFM, IFM, slope, weighted score)
├── <fusion>.py               # Decision-fusion state machine
├── <calibration>.py          # Offline calibration of windows and thresholds
├── <simulation>.py           # SimGlucose scenarios (meals, boluses, pump failures)
└── README.md
```

## Requirements

- Python 3.x
- [simglucose](https://github.com/jxx123/simglucose) (UVa/Padova simulator)
- numpy, pandas, matplotlib

```bash
pip install simglucose numpy pandas matplotlib
```

## Usage

<!-- Replace with your real commands -->

```bash
# 1. Offline calibration for one patient profile
python <calibration>.py --patient adolescent#001

# 2. Online detection with failures injected
python <simulation>.py --patient adolescent#001 --faults
```

## Authors and supervision

- Lyna Rouaida Larem
- Ala Yara Ettorche

Supervisors: Dr. Fateh Latreche, Ms. Rayene Chabani.


## References

- Howsmon et al., "Continuous Glucose Monitoring Enables the Detection of Losses in Infusion Set Actuation (LISAs)", *Sensors*, 2017.
- Man et al., "The UVA/PADOVA Type 1 Diabetes Simulator: New Features", *J Diabetes Sci Technol*, 2014.
- Bequette, "Fault Detection and Safety in Closed-Loop Artificial Pancreas Systems", *J Diabetes Sci Technol*, 2014.
