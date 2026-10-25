# Banc extérieur Iconclass (Emblematica) : têtes et VLM hors du jeu de test Iconclass (règles écrites avant calcul)

Date : 25/10/2026, avant tout calcul sur ce banc. Boucle « nouveau modèle ? », étape 4 ; accord Claude–Codex (tour 3 : vérité de
catalogue rattachée aux images, filtrée contre tout ICAI ; aucune étiquette fabriquée). Scripts : `scripts/build_external_iconclass_bench.py`
(candidats, fait), `scripts/remote_external_iconclass.py` (filtre, tirage, têtes), `scripts/remote_external_vlm.py` (VLM, à venir).

## Candidats (déjà calculés, sans regarder aucune prédiction)

13 428 picturae Emblematica locales avec notations de catalogue (parseur `parse_emblem_xml`). Filtre 1, hachage d'image 256 bits contre
les 87 745 images ICAI (`data/derived/external-iconclass/near256.json`) : étalonné à l'œil sur des paires (moins de 45 bits = même
emblème numérisé deux fois ; une copie vue à 52) ; tout livre ayant une picture à moins de 50 bits d'une image ICAI est retiré entier.
Restent 7 320 picturae de 89 livres.

## Filtre 2, puis tirage (dans `remote_external_iconclass.py`)

- Caractéristiques C-RADIOv3-L identiques à la tête (`FEATURE_CONFIG`), comparées aux 87 004 caractéristiques ICAI (train + test du
  calcul complet). Une picture dont le plus proche voisin ICAI a un cosinus ≥ 0,90 fait retirer tout son livre. Publiés : histogramme des
  cosinus, livres retirés, et une planche des 12 paires les plus proches restantes (regardée avant toute évaluation).
- Tirage (graine 0) parmi les livres restants : chaque livre va entier en développement ou en test ; au plus 15 picturae par livre ;
  développement ≈ 200 picturae, test ≈ 800. Listes gelées dans `results/iconclass-externe/` avant toute prédiction.

## Vérité et mesures

- Vérité principale : notations de catalogue **sans le chapitre 86** (devises et proverbes écrits, souvent `86(TEXTE)` : texte, pas image) ;
  vérité secondaire : toutes les notations. Notations illisibles comptées et publiées.
- Mêmes mesures que `results/iconclass-tete/` : F1 hiérarchique par la notation (principal), codes exacts, intervalles bootstrap appariés.
- Ce sont des annotations de catalogue, incomplètes : on mesure l'accord avec le catalogue, pas la vérité exhaustive de l'image.

## Systèmes

- T et K **tels qu'appris** dans `results/iconclass-tete/` (mêmes poids, seuils, voisins), sans aucun réglage sur ce banc.
- 4B sft-brillfull (réglages du fichier publié), et 9B sft-brillfull si un GPU de 24 Go est disponible (Colab L4), sur le **test** seulement.
- Le développement ne sert qu'à vérifier le code ; aucun réglage n'y est choisi.

## Règle (exploratoire)

- **Transfert** : pour chaque système, écart entre son F1 sur les 788 d'ICAI et son F1 sur ce test ; publié sans seuil.
- **Brique retenue pour les codes** : le système au meilleur F1 principal sur ce test, sauf si un autre est à moins de 2 points (borne basse
  de l'écart apparié ≥ −2 points) **et** au moins 10 fois moins cher ; alors celui-ci. Les têtes T et K ne sont pas réentraînées ici.

## Amendements du 25/10/2026, avant tout calcul (relecture Codex, rapport privé)

1. **Projection commune** : la mesure principale retire le chapitre 86 de la vérité **et** des prédictions ; les images sans vérité
   restante sont hors du dénominateur principal (nombre publié, le même pour tous les systèmes). Le transfert ICAI → extérieur est
   calculé sous la même projection, en re-scorant aussi les 788 d'ICAI ainsi.
2. **Deux phases séparées** : `prepare` (filtre 2, livres retirés persistés, caractéristiques en cache, tirage, `frozen.json` avec les
   empreintes des listes, du filtre, du cache et de la tête) puis arrêt ; planche des 12 paires restantes les plus proches regardée sur
   le Mac (images ICAI locales) ; `vu.json` porte l'empreinte de `frozen.json` ; `evaluate` refuse de tourner sans lui ni si une
   empreinte diffère.
3. **T et K évalués tous les deux**, tels qu'appris (K : étiquettes du train relues dans le jeu épinglé, même partage de validation par
   groupes d'images identiques, graine 0, même seuil).
4. **Comparaisons** : bootstrap apparié **par livres** (les livres sont tirés, pas les images) ; latence comparée sur un lot commun.
5. Le calcul 64 bits du constructeur est abandonné ; `build_external_iconclass_bench.py` lit désormais le hachage 256 bits, seuil 50 par
   livre, étalonnage archivé dans `results/iconclass-externe/etalonnage/` ; il redonne 7 320 picturae, 89 livres.
6. Le 4B relancé n'est pas déclaré équivalent à ses sorties publiées (−0,37 point, IC95 [−3,76 ; +3,35] sur 50 images, fidélité 62 %) :
   sur le banc extérieur, seules ses nouvelles sorties comptent.

## Amendement 7 (25/10/2026, après `prepare`, AVANT toute prédiction sur ce banc)

La phase `prepare` (dossier Colab `/content/extout`, gel `results/iconclass-externe/gel/`) a retiré 56 livres sur 89 au seuil de cosinus
0,90 et ne laissait que 192 images de test (cible 800). Étalonnage à l'œil sur des paires tirées au hasard dans chaque tranche de
cosinus (`results/iconclass-externe/etalonnage/cos_*.jpg`, 8 paires par tranche, graine 0) :
- **≥ 0,95** : vraies copies ou autres états/éditions du même emblème (cœurs, tour, devise) ;
- **0,90-0,95** : presque toujours des emblèmes différents, rapprochés par la mise en page de pages entières numérisées (planches
  doubles ou triples, marges, mire de couleur).
Nouveau seuil : un livre est retiré si l'une de ses picturae a un cosinus **≥ 0,95** avec une image ICAI (7 livres retirés ; 5 544
picturae et 82 livres restent). Le premier gel (`gel/`) est abandonné sans évaluation ; `prepare` est relancé dans un dossier neuf
(`/content/extout2`), mêmes règles par ailleurs. Aucune prédiction n'a été produite ni regardée sur ce banc.
Remarque : beaucoup de picturae Emblematica sont des pages entières ; la tête rapproche aussi les mises en page.
