# Pose et parties contre le vecteur global, oracle Xanto : résultat (règle : `PREINSCRIPTION.md`, amendée)

25/10/2026. Kaggle T4, `mabaku/caypollard-xanto-parts-run` (version 2), 81 min. B0 reproduit exactement sur les 46 requêtes qui ont une
découpe oracle (`kaggle/out/b0_controle.json`). Planche regardée avant les chiffres (`kaggle/out/planche.png`) : squelettes plausibles
sur les figures isolées debout (Q32, Q34), désordonnés sur les groupes et les corps couchés (Q04, Q24), et une figure de plat trop
petite (Q40) ; les boîtes de parties se recouvrent beaucoup. Détail par requête : `kaggle/out/rangs.csv`.

## Verdict préinscrit : aucune variante ne gagne

| 48 requêtes, 6 891 découpes | rang 1 | 10 premières | 50 premières | rang médian | monte / descend contre B0 | p (signe) |
|---|---|---|---|---|---|---|
| B0, vecteur global DINOv3 | 5 | 9 | 22 | 78 | — | — |
| P, pose seule | 3 | 11 | 15 | 510 | 19 / 25 | 0,45 |
| **F, vecteur global + pose** | 3 | **17** | 23 | 63 | 23 / 20 | 0,76 |
| K, parties | 5 | 11 | 18 | 135 | 20 / 23 | 0,76 |
| KP, parties + pose | 5 | 15 | 20 | 129 | 22 / 22 | 1 |

Il fallait au moins 18 requêtes dans les 10 premières et un test du signe < 0,0125. F en a 17, et son test du signe sur les rangs n'est
pas significatif : la règle n'est pas remplie.

## Descriptif (sans verdict)

- **La fusion avec la pose fait entrer 9 requêtes dans les 10 premières et n'en fait sortir qu'une** (entrées : Q06, Q07, Q12, Q15,
  Q23, Q33, Q42, Q45, Q51 ; sortie : Q18). Ce comptage d'entrées et de sorties n'était pas préinscrit comme critère : c'est un signal,
  pas une preuve. Le test préinscrit porte sur tous les rangs, où la pose fait aussi beaucoup descendre (rang médian de P : 510).
- **La pose seule ne suffit pas** (rang médian 510 contre 78) : comme chez Jenicek & Chum, elle aide quand elle s'ajoute à
  l'apparence, pas à sa place.
- **Les parties apportent peu** (K : 11 contre 9 ; KP : 15) : les figures du plat sont petites, et les boîtes de parties sont larges.
- 424 découpes de la galerie sur 6 891 n'ont pas six articulations ; groupes et compositions restent difficiles pour toutes les variantes.

## Conséquences

- La comparaison partie par partie, telle qu'elle est construite ici, n'est pas une piste prioritaire.
- **La fusion apparence + pose mérite un test confirmatoire** sur un lot neuf de paires (nouveau lot fermé, règle écrite avant), comme
  brique nommée « F » ; le seuil « ≥ 18 » et le test du signe restent ceux de la règle.
- Le vecteur global reste le principal goulot (oracle : 9/48) : le passage gravure → peinture n'est toujours pas résolu.
