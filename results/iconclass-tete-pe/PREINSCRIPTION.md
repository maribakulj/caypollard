# Tête Iconclass sur PE-Spatial (variante ouverte) : règles écrites avant calcul

Date : 25/10/2026, avant tout calcul. Demande de Marcel : « lance le chantier PE-Spatial pour voir », avec relecture de Codex avant
chaque lancement. Brique nommée « T-PE » / « K-PE », distincte de la tête C-RADIO (`results/iconclass-tete/`), dont rien ne change.

## Pourquoi

C-RADIOv3-L est sous licence NVIDIA Open Model License : permissive (usage commercial, dérivés et sorties autorisés), mais révocable et non
libre au sens strict. PE-Spatial-L (Meta, `timm/vit_pe_spatial_large_patch14_448.fb`, Apache 2.0) permettrait une version entièrement
libre. Question : que coûte ce changement d'encodeur en qualité ?

## Ce qui change, et seulement cela

- Encodeur : PE-Spatial-L, révision `ac6f424680a305470340dac75a0a9b4c47ee030f`, poids `model.safetensors` à cette révision ; image complétée
  en carré au gris 128, 448 px bicubique, normalisation du modèle (moyenne et écart 0,5), autocast fp16 ; vecteur = moyenne des jetons de
  patch (le regroupement par défaut du modèle, `global_pool = avg`), normalisé (1 024 dimensions ; C-RADIO : 4 096).
- Tout le reste est identique et vient des mêmes scripts (`scripts/remote_iconclass_head.py` avec `CAY_ENCODER=pe-spatial`) : données et
  révision, précontrôle, partage de validation, vocabulaire, tête, perte, grille de seuils, époques, mesures, latence.
- Banc extérieur : **mêmes listes gelées** que pour C-RADIO (`results/iconclass-externe/gel2/`, regardées et signées), sans nouveau
  filtre ni nouveau tirage (`remote_external_iconclass.py evaluate-on`), pour que les deux têtes voient exactement les mêmes images.

## Mesures et règle (descriptives, pas de remplacement automatique)

- Mêmes mesures que la tête C-RADIO : F1 hiérarchique par la notation (ICAI 788 ; Emblematica 762 sans chapitre 86), codes exacts,
  latence de bout en bout.
- Comparaison appariée T-PE − T (C-RADIO) : bootstrap par images sur ICAI (2 000 tirages, graine 0), par livres sur Emblematica.
- **Lecture fixée d'avance** (amendée le 25/10 à la demande de Marcel, avant tout calcul) : **les deux têtes sont publiées**, chacune
  avec sa licence (C-RADIO : MIT pour notre code et nos poids, plus la licence NVIDIA et le fichier NOTICE requis ; PE-Spatial : Apache 2.0).
  La règle ne décide que de la version **recommandée par défaut** : PE-Spatial si la borne basse de l'écart T-PE − T est ≥ −2 points sur
  les deux bancs, sinon C-RADIO. Dans les deux cas, T-PE est aussi comparée aux VLM publiés (mêmes chiffres de référence).

## Calcul

Colab T4, une cellule chaînée (encodage + tête, puis `evaluate-on`), ~2 h, ~3 unités ; total de la boucle précédente : 7,4 unités.
Essai à blanc d'abord (`CAY_SMOKE=64`), après relecture par Codex.

## Comparaison par usage (ajoutée le 25/10 à la demande de Marcel, avant tout calcul)

Le choix peut dépendre de l'usage ; on publie, pour T (C-RADIO), T-PE, et les VLM 4B et 9B, sans seuil de verdict :
- **Par branche d'Iconclass** (le premier chiffre de la notation : 1 religion et magie, 2 nature, 3 être humain, 4 société et objets,
  5 idées abstraites, 6 histoire, 7 Bible, 8 littérature, 9 mythologie) **et deux sous-branches fixées** : 31A (corps, postures, gestes)
  et 33 (relations entre personnes). Pour une branche b, on garde les images dont la vérité contient un code de b, et l'on calcule le F1
  hiérarchique entre les ensembles d'ancêtres restreints aux codes de b (vérité et prédiction) ; nombre d'images par branche publié ;
  écart apparié T-PE − T avec bootstrap (images pour ICAI, livres pour Emblematica). Les deux bancs, avec la projection sans 86 pour
  Emblematica.
- **Recherche d'images semblables** (sujets, copies, œuvres dérivées, relations) : chiffres déjà mesurés des vecteurs C-RADIO et
  PE-Spatial dans `results/atelier/TABLEAU.md` (aucun nouveau calcul) ; rappelés à côté.
- **Coût** : latence de bout en bout et mémoire GPU de chaque tête, même T4.
Calcul local et léger à partir des prédictions sauvegardées (`scripts/iconclass_par_usage.py`, écrit et relu avant d'ouvrir les
résultats de T-PE).
