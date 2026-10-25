# Tête Iconclass sur PE-Spatial (T-PE) : résultat

25/10/2026. Règles écrites avant calcul : `PREINSCRIPTION.md` (relue par Codex). Calcul sur Kaggle T4 (kernel
`mabaku/caypollard-iconclass-pe-run`, après la perte d'une session Colab), même script que la tête C-RADIO
(`scripts/remote_iconclass_head.py`, sha256 `ad7cb459…`, `CAY_ENCODER=pe-spatial`), puis `remote_external_iconclass.py evaluate-on`
sur les listes gelées gel2 (`externe/reference.json` : même tirage, mêmes images, empreintes vérifiées). Chaîne : encodage + tête
2 h 14, banc extérieur 9 min (`full/chain.log`).

## Verdict de la règle

| | T (C-RADIO) | T-PE | T-PE − T [IC95] |
|---|---|---|---|
| ICAI 788, F1 hiérarchique | 54,8 % | 46,7 % | −8,0 [−9,2 ; −6,9] |
| Emblematica 762 (sans 86), F1 hiérarchique | 26,1 % | 19,2 % | −6,9 [−8,1 ; −5,6] (bootstrap par livres) |

La borne basse est sous −2 points sur les deux bancs : **la version recommandée par défaut reste C-RADIO.** Les deux têtes sont
publiées comme prévu, T-PE comme variante entièrement libre (Apache 2.0) et nettement moins bonne.

T-PE face aux VLM : sur ICAI elle dépasse le 4B (+1,9 [+0,5 ; +3,4]) et reste sous le 9B (−1,6 [−3,1 ; +0,0]) ; sur Emblematica elle
reste devant les deux (4B 14,6 %, 9B 15,8 %).

## Autres mesures (ICAI 788)

| | T (C-RADIO) | T-PE |
|---|---|---|
| codes exacts, F1 | 0,275 | 0,203 |
| codes par image | 5,0 | 6,3 |
| voisins K (même encodeur), F1 hiérarchique / codes exacts | 45,0 % / 0,356 | 40,9 % / 0,325 |
| latence de bout en bout, lot 1, T4 (médiane) | 0,058 s | 0,074 s |

Le choix sur validation désigne T (la tête) dans les deux cas. Époque retenue pour T-PE : la dernière (14 sur 0-14), la validation
montait encore (0,473) ; les 15 époques sont la règle commune, rien n'a été prolongé.

## Par usage (`par_usage.json`, `scripts/iconclass_par_usage.py`, relu par Codex avant l'ouverture des résultats)

Descriptif, sans seuil de verdict ; IC95 seulement à partir de 20 images.

- **ICAI** : T-PE est en dessous de T dans **toutes** les branches (de −2,8 à −11,7 points ; mythologie 9 : −11,7).
- **Emblematica** : T-PE est en dessous dans la plupart des branches, souvent de beaucoup (religion 1 : −13,4 ; Bible 7 : −12,9 ;
  mythologie 9 : −26,6), à égalité sur l'être humain 3 et les idées abstraites 5 (presque nulles pour tous).
- **Deux branches où T-PE semble gagner, à ne pas lire comme un avantage** : relations entre personnes 33 (34,6 % contre 1,0 %) et
  corps-gestes 31A (18,1 % contre 12,2 %). Regardé avant de l'écrire : T-PE met un code 33 sur **323 des 800** images du test
  (`33C2` sur 215), contre 10 pour T, alors que 50 images seulement portent un code 33 dans la vérité ; même chose pour 31A (389
  images contre 206). La mesure par branche ne garde que les images dont la vérité a un code de la branche, donc ne compte pas ces
  fausses alarmes ailleurs : le gain apparent vient d'une prédiction distribuée presque partout, que la mesure globale pénalise.
- **Recherche d'images semblables** (chiffres déjà mesurés, `results/atelier/TABLEAU.md`, premier score de chaque banc) : le vecteur
  C-RADIO devance nettement le vecteur PE-Spatial partout (copies 0,94 contre 0,35 ; même sujet 0,92 contre 0,25 ; œuvres dérivées
  0,84 contre 0,42 ; relations 0,60 contre 0,32).
- **Coût** : T-PE un peu plus lente (0,074 s contre 0,058 s par image) ; vecteur 1 024 dimensions contre 4 096.

Aucun usage mesuré où PE-Spatial l'emporte. Son seul avantage est la licence.

## Fichiers

- `full/` : `resultats.json`, `predictions.json`, `run.json`, `preflight.json`, `journal.jsonl`, `chain.log`.
- `externe/` : `resultats.json`, `predictions.json`, `reference.json`.
- `par_usage.json`.
- Poids `head.pt` de T-PE (73 Mo) : hors git, sur le Drive (`caypollard-archives/tete-iconclass-pe/`).
