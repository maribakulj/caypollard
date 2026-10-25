# Banc extérieur Iconclass (Emblematica) : résultat (règle : `PREINSCRIPTION.md`, amendements 1-7)

25/10/2026. Gel : `gel2/` (seuil de copie 0,95 ; 5 544 picturae, 82 livres ; dev 202, test 800 ; planche des 12 voisins restants
regardée, `gel2/planche_12_voisins.jpg`, aucune copie ; `vu.json`). Tête réentraînée sur Colab = reproduction exacte du premier calcul
(T 54,79 %, K 45,04 % sur les 788 d'ICAI). Mesure principale : F1 hiérarchique par la notation, chapitre 86 retiré de la vérité et des
prédictions ; images sans vérité restante exclues (38 sur 800 au test).

| Système | ICAI 788 (même projection) | Emblematica test (762 images, 58 livres) | Chute | Temps par image |
|---|---|---|---|---|
| **T, tête sur C-RADIO** | 54,6 % | **26,1 %** | −28,5 | 0,058 s (T4) |
| K, plus proches voisins | 45,2 % | 8,2 % | −37,0 | ~0,06 s |
| Qwen3.5 4B sft-brillfull | 44,7 % (publié) | 14,6 % (relancé, 800/800 JSON valides) | −30,1 | 2,16 s (T4) |
| Qwen3.5 9B sft-brillfull | 48,2 % (publié) | 15,8 % (relancé sur L4, 800/800 JSON valides) | −32,4 | 2,05 s (L4) |

- **T − 4B sur le test extérieur : +11,5 points, IC95 [9,7 ; 13,3] ; T − 9B : +10,3 points, IC95 [7,9 ; 12,7]** (bootstrap apparié par
  livres, 2 000 tirages, 58 livres).
- Codes exacts : F1 ≈ 0,04 pour T ; tous les systèmes trouvent surtout des ancêtres (catégories larges), rarement le code précis.
- Toutes notations comprises : T 23,9 %.
- Développement (202 images) : T 26,2 %, K 9,9 % — cohérent avec le test.

## Lecture

- **Tous les systèmes chutent fortement hors du jeu Iconclass** : l'avance mesurée sur les 788 était en partie propre à ce jeu
  (mêmes sources, mêmes catalogueurs, peut-être mêmes conventions). Emblematica annote d'autres choses, plus finement, et beaucoup de ses
  images sont des pages entières.
- **Verdict préinscrit : la tête est la brique retenue pour les codes** — meilleur F1 principal sur ce test, devant le 9B et le 4B,
  pour un coût environ 35 fois plus faible.
- K s'effondre : voter les codes des voisins ne marche que si des images très semblables existent dans l'entraînement.
