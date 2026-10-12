# Système de détection de pannes - Pancréas Artificiel

## Description
Système hybride de détection des pannes de pompe à insuline
combinant un filtre de Kalman et l'approche LISA.

## Structure
- offline/ : Calibration des paramètres (phase préparatoire)
- online/  : Détection en temps réel + métriques

## Prérequis
pip install simglucose numpy matplotlib requests

## Utilisation
1. Lancer d'abord le script offline pour calibrer les paramètres
2. Copier les paramètres obtenus dans le script online correspondant
3. Lancer le script online pour la détection en temps réel

## Profils patients
- Enfant   : child#001
- Adolescent : adolescent#001
- Adulte   : adult#001