# Codes Iconclass : tête sur C-RADIO contre les VLM publiés — résultat (règle : `PREINSCRIPTION.md`, amendée)

25/10/2026. Colab T4 (session `cayhead`), 1 h 44 de calcul (encodage des 87 004 images, 15 époques), environ 3 unités. Précontrôle
passé (références sft4b et sft9b alignées sur le test épinglé). Sorties : `full/` (résultats, prédictions, journal ; poids `head.pt`
gardé hors git).

## Verdict préinscrit : la tête dépasse les deux VLM, à un coût environ 36 fois plus faible

| 788 images du test (partage de l'auteur) | F1 hiérarchique (notation) | codes exacts : précision / rappel / F1 | codes par image |
|---|---|---|---|
| **T, tête apprise (choisie sur la validation)** | **54,8 %** | 0,258 / 0,293 / 0,275 | 4,98 |
| K, plus proches voisins | 45,0 % | 0,334 / 0,380 / 0,356 | 5,00 |
| Qwen3.5 9B sft-brillfull (prédictions publiées) | 48,3 % | 0,299 / 0,234 / 0,262 | 3,44 |
| Qwen3.5 4B sft-brillfull (prédictions publiées) | 44,8 % | 0,257 / 0,201 / 0,226 | 3,43 |
| 4B ancien (contaminé, hors verdict) | 41,9 % | — | 2,79 |
| codes les plus fréquents | 15,4 % | — | 4 |

- **Écart apparié (bootstrap, 2 000 tirages)** : T − 4B = +10,0 points [8,6 ; 11,4] ; T − 9B = +6,5 points [5,1 ; 7,9]. Les deux bornes
  basses sont au-dessus des seuils (≥ −2 contre le 4B ; > 0 contre le 9B).
- **Coût** : tête 0,058 s par image (lot 1, image → codes, T4) contre 2,12 s pour le 4B relancé sur la même T4 (`colab/vlm_bench.json`) :
  environ 36 fois moins. Selon la règle, la tête **remplace les VLM comme candidate** pour le banc extérieur.

## Descriptif et limites (sans verdict)

- Validation : K 47,7 %, T 55,2 % ; meilleure époque 14 (la dernière) et seuil 0,60, **borne haute de la grille fixée d'avance** : un
  entraînement plus long ou un seuil plus haut pourraient faire mieux, mais ce serait un réglage choisi après coup (autre brique).
- La tête propose plus de codes que les VLM (5 contre 3,4) ; elle gagne aussi sur les codes exacts (F1 0,275 contre 0,262 pour le 9B),
  et K, plus simple, est meilleur encore sur les codes exacts (0,356) mais moins bon en F1 hiérarchique.
- 10 % des codes du test (349, dans 183 images) sont hors du vocabulaire de la tête (codes vus moins de 5 fois) : comptés comme manqués.
- 3 images du test ont une copie exacte dans le train (signalé, partage de l'auteur conservé).
- Mesure : F1 hiérarchique « par la notation » (simplification déclarée), pas le H-F1 de l'auteur (ses valeurs déclarées : 4B 52,2 ;
  9B 56,1). Le 4B relancé n'est pas identique à ses sorties publiées (fidélité 62 %) mais de même qualité sur 50 images.
- Lot déjà utilisé par l'auteur pour ses choix : verdict exploratoire. La généralisation est jugée sur le banc extérieur (Emblematica).
