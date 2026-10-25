# Codes Iconclass : encodeur gelé + tête, contre les VLM Iconclass publiés (règles écrites avant calcul)

Date : 25/10/2026, avant tout calcul. Boucle « nouveau modèle ? » (`~/Documents/reviews/caypollard-boucle-modele.md`), étapes 2 et 5,
accord Claude–Codex (`caypollard-notre-modele-2026-10-25.md`, tours 1-3). Script : `scripts/remote_iconclass_head.py` (Kaggle, T4).

## Question

Une petite tête apprise sur un encodeur d'images gelé (C-RADIOv3-L) prédit-elle les codes Iconclass aussi bien que les VLM
ajustés de van Strien (Qwen3.5 4B et 9B « sft-brillfull »), et à quel coût ?

## Données (figées)

`davanstrien/iconclass-vlm-brillfull`, révision 8045366ff092a06e8e1c4e47f1eb43cb2a54d732 (CC0) : train 86 216, test 788 (partage de
l'auteur par hachage du nom de fichier). Le test a déjà servi aux choix de l'auteur : banc de comparaison, pas preuve finale.

## Pourquoi on ne relance pas leurs modèles ici

Le code de leur H-F1 est introuvable (dépôt privé), et leurs fiches se contredisent (45,3 sur la fiche 4B ; 52,2 dans le fichier public
`passk/sft4b_1ep.json` pour le même modèle avec d'autres réglages). En revanche, `davanstrien/iconclass-eval-preds` publie, image par
image, leurs **prédictions gloutonnes** sur les 788 images. On applique donc **la même mesure, la nôtre, définie ci-dessous, à leurs
prédictions publiées et aux nôtres**. Leurs H-F1 déclarés sont recopiés à côté, sans valeur de verdict. Fichiers utilisés (et leur
empreinte SHA-256 journalisée) : `passk/sft4b_1ep.json`, `passk/sft9b_1ep.json` (références), `passk/sft4b_orig.json` (ancien
checkpoint, contaminé selon l'auteur : publié à part), `internvl35_4b_sft_brillfull_1ep_test788.json` si alignable. Contrôle : leurs
codes vérité doivent être identiques aux étiquettes du test, image par image (sinon arrêt).

## Mesures

- **F1 hiérarchique par ensembles (principal)** : chaque code est remplacé par lui et tous ses ancêtres (chiffre, chiffre, lettre(s),
  puis chaque chiffre ; parenthèses et clés retirées pour la hiérarchie) ; F1 entre l'ensemble prédit et l'ensemble vrai, par image,
  moyenné sur les 788 ; prédiction vide ou JSON invalide = 0.
- **Codes exacts** : précision, rappel et F1 micro sur les chaînes complètes (parenthèses comprises).
- Publiés aussi : nombre et profondeur des codes proposés ; baseline « codes les plus fréquents du train » ; leurs H-F1 déclarés ;
  intervalles par bootstrap sur les images (2 000 tirages, graine 0) et écart apparié avec chaque référence.

## Nos systèmes (tous fixés ici)

Caractéristiques : C-RADIOv3-L, image complétée en carré puis 448 px, jeton de synthèse et moyenne des caractéristiques spatiales,
concaténés et normalisés. Validation : 5 % du train tirés au hasard (graine 0) ; le partage de l'auteur ne donne ni livre ni œuvre :
le regroupement par provenance est impossible ici, ce qui est déclaré.
- **K, plus proches voisins** : 20 voisins du train (cosinus) ; un code est proposé si sa part de votes pondérés dépasse un seuil choisi
  sur la validation (grille 0,05-0,60).
- **T, tête apprise** : MLP (une couche cachée de 2 048, dropout 0,2) sur les codes vus au moins 5 fois dans le train et tous leurs
  ancêtres ; perte asymétrique (gamma négatif 4, positif 0, marge 0,05) ; 15 époques, AdamW 1e-3 ; époque et seuil choisis sur la
  validation (F1 hiérarchique). Sortie : codes au-dessus du seuil, sans ceux qui sont ancêtres d'un autre code proposé. Codes du test
  hors vocabulaire : comptés dans le score principal, part publiée.

## Règle

- **La tête vaut d'être développée** si le meilleur de K et T (choisi sur la validation, pas sur le test) a un F1 hiérarchique au moins
  égal à celui du 4B publié moins 2 points (borne basse de l'écart apparié bootstrap ≥ −2 points), pour un coût d'inférence au moins
  10 fois plus faible (temps mesuré sur T4, prétraitement compris, contre 4B mesuré sur le même GPU sur 50 images).
- **Elle remplace les VLM** si elle dépasse aussi le 9B publié (borne basse de l'écart > 0).
- Sinon : on adopte le VLM comme brique de codes, et la tête n'est pas poursuivie sous cette forme.

## Contrôle de vitesse et de fidélité du 4B

Le 4B (révision 597a5f5c756c21f372bf73d4fa5abcfb745daecb, fp16, réglages du fichier publié : glouton, 768 jetons, pénalité de
répétition 1,1, prompt d'entraînement) est relancé sur 50 images du test tirées au sort (graine 0) : temps par image, et part des
sorties identiques aux sorties publiées (si < 80 %, les prédictions publiées ne sont plus considérées comme reproductibles et on le dit).

## Amendements du 25/10/2026, avant tout calcul complet (relecture Codex : `~/Documents/reviews/caypollard-iconclass-tete-codex-2026-10-25.md`)

1. **Hiérarchie lue dans la notation** (mesure nommée « F1 hiérarchique par la notation ») : une notation est faite de parties jointes
   par `:` ; chaque partie se lit en jetons (chacun des deux premiers chiffres, une lettre ou une lettre doublée, chaque chiffre, puis
   chaque parenthèse — nom `(NOM)` ou clé `(+..)` — comme un jeton, et les chiffres qui la suivent un à un). Ancêtres = préfixes finissant
   sur un jeton ; un composé garde sa chaîne et les ancêtres de chacune de ses parties. Une notation illisible ne compte que pour
   elle-même (nombre publié). La même fonction sert aux cibles, au vocabulaire, à la sortie (« le plus spécifique ») et à la mesure.
   C'est une simplification déclarée : elle ne lit pas l'arbre Iconclass officiel (une lettre doublée a pour parent la division).
2. **Précontrôle avant tout encodage** : références obligatoires sft4b_1ep et sft9b_1ep présentes, indices exactement 0…787 et uniques,
   étiquettes identiques au test épinglé ligne par ligne ; sinon arrêt. InternVL retiré (schéma différent, facultatif). sft4b_orig
   publié à part, « contaminé, hors verdict ».
3. **Caches** : C-RADIOv3-L et son processeur épinglés (révision 9d0413465e8a91e67bbf2c1ad342815478d1b906) ; chaque fichier de
   caractéristiques a un manifeste (jeu et révision, fichier, nombre de lignes, mode essai, configuration, empreinte des étiquettes)
   qui doit être identique pour être repris ; écriture atomique ; comptes 86 216 / 788 vérifiés hors essai.
4. **Validation** : 5 % du train tirés par groupes d'images identiques à l'octet (graine 0) ; copies exactes train/test comptées.
   Graines torch fixées ; meilleure tête sauvegardée à chaque amélioration ; journal par époque.
5. **Perte** : tout code absent est traité comme négatif (l'incomplétude des catalogues n'est pas masquée) — hypothèse déclarée.
6. **Coût** : latence de bout en bout (image → codes) du système choisi, lot de 1, après échauffement, sur 50 images du test (graine 0),
   même GPU. Le verdict « 10× moins cher » attend la mesure du 4B dans les mêmes conditions (script séparé, relu avant envoi) ; la
   fidélité compare les ensembles de codes et le statut JSON.
7. **Portée** : verdict exploratoire sur un lot déjà utilisé par l'auteur ; « remplacer » veut dire « candidat pour le banc extérieur ».
