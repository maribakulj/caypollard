# Graphes de scène et recherche en deux temps : revue de sources

Date : 25/10/2026. Question : une première passe structurelle (figures détectées, poses, rôles, relations) peut-elle dire à un apparieur coûteux (SegSwap, patches DINOv2 / PE-Spatial, appariement local) OÙ regarder, sur des milliers à des millions d'images d'art ?

Convention : « OUVERT » = j'ai ouvert la page ou le PDF et lu le texte (ou un résumé produit par l'outil de lecture sur cette page). « RECHERCHE SEULE » = vu uniquement dans un résultat de moteur de recherche, non ouvert. « NON VÉRIFIÉ » = je n'ai pas pu lire. Les chiffres sont ceux des auteurs, non reproduits.

## Réponse courte

1. Oui pour la pose + vérification géométrique, avec un précédent direct sur des peintures : Jenicek & Chum 2019. Première passe = distance de pose entre figures (invariante au miroir) ; seconde = RANSAC sur les points clés, qui aligne toutes les figures à la fois. Sans la seconde passe, la première est PIRE que les deux baselines globales.
2. Les graphes de scène « sémantiques » (sujet-prédicat-objet) aident peu pour notre usage : le prédicat apporte presque rien, la génération du graphe coûte cher et se trompe. Les chiffres utiles viennent des travaux qui gardent des descripteurs visuels par objet.
3. Les deux temps global puis local sont la norme à grande échelle (D2R, RRT, AMES, Ufer et al. sur l'art). Mais ces travaux filtrent par similarité visuelle, pas par structure. Aucune source lue ne montre une première passe « graphe de figures » guidant SegSwap ou DINOv2 dense : c'est une extrapolation à tester, pas un résultat établi.
4. SegSwap compare des PAIRES ; ses auteurs écrivent eux-mêmes que c'est difficile à grande échelle. Il lui faut donc impérativement un étage de présélection.

## 1. Graphes de scène pour la recherche d'images

### Johnson et al., CVPR 2015
- Page CVF (https://openaccess.thecvf.com/content_cvpr_2015/html/Johnson_Image_Retrieval_Using_2015_CVPR_paper.html) : 403, NON OUVERT. PDF CVF idem 403. Extended abstract (https://www.cv-foundation.org/openaccess/content_cvpr_2015/ext/2B_036_ext.pdf) : ouvert, mais c'est un autre papier (« Real-World Scene Graphs Dataset »), sans résultats lisibles.
- Ce que je sais par recherche seule (résultat de moteur, https://www.researchgate.net/publication/308871896_Image_retrieval_using_scene_graphs, non ouvert) : 5 000 graphes de scène annotés à la main sur des images YFCC100m∩COCO ; requête = graphe ou sous-graphe ; le modèle (champ aléatoire conditionnel sur les groundings) bat les méthodes à objets seuls ou à caractéristiques bas niveau. Chiffres de recall : NON VÉRIFIÉS. Un papier ultérieur (Schuster et al., via la même recherche) cite Recall@10 de 27,1 % avec graphes analysés contre 33,4 % avec graphes humains : RECHERCHE SEULE, donc l'écart humain/automatique est de l'ordre de 6 points.
- Étage 1 : aucun (c'est un classement direct par CRF, sans index). Pas de leçon d'échelle.

### Schroeder & Tripathi, CVPRW 2020 (OUVERT, https://arxiv.org/pdf/2005.06653 ; résumé https://arxiv.org/abs/2005.06653)
- Requête = triplet <sujet, prédicat, objet> ; embedding appris à partir du graphe seul (pas de pixels), base de 3 100 relations visuelles tirées de COCO-Stuff, graphes SYNTHÉTIQUES (relations géométriques : gauche de, au-dessus de, etc.).
- Recall@100 : 46 % (triplet, sujet+objet) ; 42 % avec le prédicat ; 29 % sujet seul ou objet seul ; prédicat seul 3 % (hasard 1 %). Le prédicat NE sert presque à rien et l'omettre améliore même de 4 points. Les auteurs rappellent que la détection de graphes de scène est sous 35 % de recall@100 sur Visual Genome (citant un état de l'art de l'époque).
- Leçon : une relation fine entre figures est le maillon faible ; les types d'objets portent le signal.

### PRISm, 2025 (OUVERT, https://arxiv.org/html/2512.18407)
- Base PSG : 47 874 images, graphes humains ; Flickr30k : 12 000 images, graphes générés par PSGTR. Test = chaque image du jeu de test sert de requête.
- PSG : NDCG@1 88,5 contre 81,3 pour CLIP (meilleure baseline) ; MAP@1 91,6 contre 79,0. Avec graphes générés (Flickr), gain plus faible, scores plus bas à cause de graphes bruités.
- Graphe seul sans vision : NDCG@1 74,1, SOUS CLIP. Graphe avec attributs visuels par objet, sans flux global : MAP@1 82,1, au-dessus de CLIP (79,0). Le meilleur est la combinaison.
- Coût : la génération du graphe fait ~94 % de la latence (~0,5 s au total), la recherche elle-même ~33 ms. Galeries de quelques milliers d'images seulement : « l'échelle n'est pas vérifiée ». La pertinence est jugée par légende de substitution, pas par humains.
- Leçon : le graphe aide à condition d'embarquer des descripteurs visuels par nœud ; son coût est dans l'extraction, qu'on ferait de toute façon pour détecter les figures.

### SPAN, 2023 (résumé seul OUVERT, https://arxiv.org/abs/2304.00590)
- Apprentissage contrastif graphe-image (Visual Genome, Open Images), propose R-Precision pour évaluer la génération de graphes par la recherche. Aucun chiffre lisible sur la page. NON VÉRIFIÉ pour les résultats.

### Graphes contre embeddings globaux : où ça échoue
- Les modèles vision-langage globaux se comportent comme des « sacs de mots » pour les relations et attributs (cité dans PRISm, OUVERT). CS-CLIP (RECHERCHE SEULE, https://arxiv.org/abs/2609.08242, non ouvert) : les corrections guidées par graphe ne sont pas uniformément meilleures que CLIP.
- Sources de défaillance lues : graphes générés bruités (PRISm), prédicats non discriminants (Schroeder), vocabulaire d'objets modernes inadapté à l'art (voir ci-dessous).

### Art : graphes de scène ou descriptions structurées
- DEArt (OUVERT, https://arxiv.org/abs/2211.01226) : plus de 15 000 images de peintures européennes (XIIe-XVIIIe), 69 classes d'objets dont plus de 50 propres au patrimoine, 12 poses pour les objets de forme humaine, ~80 % d'images non iconiques. Détecteurs par transfert : précision 31,2 % (chiffre vu en RECHERCHE SEULE, https://zenodo.org/records/6984525 et recherche arXiv ; non confirmé dans le résumé ouvert). Aucune relation : mon intitulé « DEArt 2025 relations » n'est pas trouvé.
- SemArt-HOI, CompoVista, ArtHOI : recherche faite, AUCUN papier correspondant trouvé. Je ne peux rien en dire. (Noms peut-être inexacts.)
- Springstein et al., poses en histoire de l'art (OUVERT, https://arxiv.org/abs/2207.02976) : jeu d'images d'art avec boîtes et points clés de figures, détection semi-supervisée meilleure que préentraînement ou transfert de style. D'après la recherche seule, un descripteur de pose de 52 dimensions (orientations entre points clés) et un test utilisateur à 12 personnes ; descripteur non confirmé sur la page ouverte.

## 2. Recherche en deux temps à grande échelle

### Detect-to-Retrieve, CVPR 2019 (OUVERT, https://arxiv.org/abs/1812.01584 ; chiffres via https://ar5iv.labs.arxiv.org/html/1812.01584)
- Étage 1 : détecteur de monuments (86 k images annotées de boîtes, 15 k monuments ; MobileNet-V2 SSD, AP@0,5 = 85,6 %) ; chaque région détectée est indexée (D2R) ou agrégée (R-ASMK, 1 descripteur par image).
- ROxford-Hard, codebook 1024 : 38,3 (image entière) -> 40,7 avec ~4,1 régions/image ; 42,0 avec agrégation. Grille de boîtes aveugle : 40,4 avec ~9 régions, soit 2,24 fois la mémoire. Système complet avec vérification spatiale : 52,4 contre 43,1 (+9,3). RParis-Hard : +1,9.
- Coût : détecteur ~27 ms/image (MobileNet-V2 SSD, GPU TitanX), 89 ms (Faster R-CNN). Index 1M distracteurs : 27,6 Go contre 10,3 Go pour la base sans régions. Latence de requête de bout en bout : non rapportée.
- Leçon : les régions DÉTECTÉES valent mieux que des régions aveugles à mémoire égale (+2 à 3 points, 2,24 fois moins de mémoire), mais le gain d'une simple indexation par régions est modeste ; l'essentiel du gain vient de la vérification spatiale. Échec cité : détections doubles, objet fondu dans la scène.

### Reranking Transformers (RRT), 2021 (OUVERT, https://arxiv.org/abs/2103.12236 ; chiffres via https://ar5iv.labs.arxiv.org/html/2103.12236)
- Étage 1 : descripteur global (DELG, https://arxiv.org/abs/2001.05027, OUVERT, sans chiffres sur la page). Étage 2 : transformeur sur les descripteurs locaux des 100 premiers.
- ROxford-Hard : 56,4 contre 54,2 pour RANSAC (+2,2) et 45,1 pour le global seul (+11,3) ; RParis-Hard : 68,6 contre 64,9 (+3,7) et 63,4 (+5,2). Avec 500 descripteurs contre 1 000 pour la vérification géométrique.
- Coût : 0,36 s/requête sur P100 (config DELG) ou 0,013 s (config SOP), 2,2 M de paramètres. Les auteurs jugent le coût encore élevé pour le très grand échelle.

### AMES, 2024 (résumé et page alphaxiv OUVERTS : https://arxiv.org/abs/2408.03282 ; https://www.alphaxiv.org/abs/2408.03282)
- Réordonnancement avec budget mémoire ~1 Ko/image ; asymétrie (plus de descripteurs côté requête, +0,3 à 1,0 point de mAP sans mémoire en plus). AMES binaire avec CVNet : ~10 Ko/image contre ~1 400 Ko pour le réordonnancement natif de CVNet ; RRT, R2Former, CVNet peuvent dépasser 1 Mo/image. Latence, nombre de candidats, mAP précis : NON VÉRIFIÉS (les PDF/HTML ont échoué).

### Recherche de régions dans l'art à grande échelle : Ufer, Lang, Ommer 2021 (PDF OUVERT, https://arxiv.org/abs/2107.06935 ; texte lu)
- Un descripteur local par patch multi-échelle (propositions en fenêtre glissante), index FAISS IVF+PQ, vote de régions. Le plus proche de « un vecteur par objet + FAISS » sur l'art.
- Brueghel (mAP) : 85,7 à 88,1 contre 79,9 pour Artminer (ancienne référence) ; LTLL 90,9 ; Oxford5K ~90.
- Coût : temps de recherche 8,5 s (5 k images) à 10,5 s (100 k images), 3 GPU P5000 ; Artminer : 12,9 min à 4,6 h. Taille d'index : 1,7 Go (5 k) à 34,2 Go (100 k), soit ~0,34 Mo/image, proportionnelle au nombre de propositions par image. À un million d'images, extrapolation linéaire ~340 Go : mon calcul, non mesuré.

### SegSwap (OUVERT : https://arxiv.org/abs/2110.15904 ; détails via https://ar5iv.labs.arxiv.org/html/2110.15904)
- Brueghel (détails d'œuvres) : 84,4 (segments COCO, transformeur) ; baselines 75,5 à 80,8. Les auteurs écrivent que l'approche doit comparer des paires d'images, ce qui rend le passage à l'échelle difficile ; la découverte sur ~900 k nœuds est quadratique (calcul de toutes les paires : ~10 h, regroupement ~2 h).
- Pour la reconnaissance de lieu, ils réordonnent les 100 premiers résultats de NetVLAD (suivant Patch-NetVLAD) : seul exemple d'usage en second étage. Temps par paire : NON rapporté.

### À propos de SuperGlue / LightGlue en réordonnancement, DINOv2 dense, PE-Spatial
- Non ouverts ni cherchés en détail : NON VÉRIFIÉ. Rien dans cette revue ne documente leurs coûts.

## 3. Recherche par pose

### Jenicek & Chum 2019, « Linking Art through Human Poses » (OUVERT en texte intégral, https://arxiv.org/abs/1907.03537 ; PDF https://arxiv.org/pdf/1907.03537)
- Étage 1 (appariement rapide) : OpenPose body-25 (25 points) ; pose normalisée par translation (racine = cou) et échelle ; distance = 1 - cosinus sur les points clés détectés dans les deux poses ; invariance au miroir explicite : q = min(distance directe, distance avec miroir). Score d'image = approximation d'appariement biparti des figures (distt) ou meilleure figure (distmin). Liste restreinte l = 50. Les figures sont comparées indépendamment : les relations entre figures n'y sont pas utilisées.
- Étage 2 : pour chaque hypothèse figure-figure, transformation (échelle isotrope, translation, miroir : 3 degrés de liberté) estimée par RANSAC, appliquée à TOUTES les figures ; score = nombre de points clés cohérents. Seuil d'inliers relatif à la taille de pose ; test d'angle du torse pour élaguer.
- Jeu : 635 images annotées par des historiens de l'art (copies et transferts de composition) ; recherche qualitative dans 37 000 images de la Web Gallery of Art.
- Résultats (précision moyenne, toutes connexions) mP@1 / mP@10 : VGG GeM 56,8 / 34,6 ; EdgeMAC 59,3 / 35,0 ; pose seule sans vérification (distt) 50,1 / 33,5 ; pose + vérification (distt) 75,3 / 50,6. Soit +16 points à mP@1 sur la meilleure baseline, mais seulement grâce à la 2e passe. Pour les seuls « transferts de composition » : mP@1 55,7 contre 41,0 (GeM) et 40,4 (EdgeMAC).
- Coût : appariement de pose 100 ms/requête sur 635 images (implémentation Python naïve, linéaire en taille de base : pas d'index) ; vérification 4,5 s/requête pour 50 candidats. Aucun index approximatif.
- Limites citées : points clés manquants ou mal associés (OpenPose), faux positifs quand une pose générique de plusieurs personnages est amplifiée par du bruit ; la détection échoue systématiquement pour l'art moderne (cubisme, impressionnisme).
- Leçons directes : (a) les figures sont comparées une à une puis groupées par géométrie commune, ce qui est exactement votre « cohérence de groupe » ; (b) le miroir est géré en prenant le minimum des deux distances, sans doubler l'index si l'on indexe les poses miroir ; (c) il faut au moins ~2/3 des points détectés.

### Autres (RECHERCHE SEULE, non ouverts)
- Vases grecs : adaptation par transfert de style (https://arxiv.org/abs/2012.05616), recherche de postures (précision/mAP rapportées, chiffres non lus).
- Illustrations d'animation : descripteur invariant à la rotation (https://arxiv.org/abs/2108.01819). Non lu.

## Pipeline visé : figures -> descripteur (pose + crop) -> FAISS -> SegSwap/dense -> cohérence de groupe

Ce que les sources soutiennent :
- Détecter puis indexer par objet est efficace et chiffré (D2R, Ufer). Le détecteur doit être entraîné pour le domaine : DEArt donne 69 classes et 12 poses sur la peinture, mais précision de détection modeste (31,2 %, non confirmée sur page ouverte).
- Pose + vérification géométrique de groupe fonctionne sur la copie et le transfert de composition (Jenicek), à condition d'une 2e passe ; la pose seule est mauvaise (mP@1 50,1 contre 56 à 59 pour les baselines globales).
- Réordonner un petit nombre de candidats (50 à 100) avec un apparieur coûteux est le schéma standard (Jenicek 50, RRT 100, SegSwap place recognition 100).
- Mémoire : un vecteur par figure reste raisonnable (D2R : 4 régions par image coûtent ~2,7 fois la mémoire de base) ; l'index patch de Ufer (~0,34 Mo/image) ne l'est pas au million sans compression.

Ce que les sources NE montrent PAS (à tester) :
- Qu'une structure de figures (poses, rôles, relations) restreigne utilement la zone que SegSwap ou le DINOv2 dense doit examiner : aucune source lue ne le fait. Seul Jenicek se rapproche, car l'alignement RANSAC donne une transformation entre les deux images, qu'on pourrait fournir à un apparieur dense comme initialisation (hypothèse mienne).
- Le gain des relations (prédicats) : plutôt faible ou nul là où mesuré (Schroeder). Garder la géométrie de groupe (positions relatives, échelle commune) plutôt que des prédicats sémantiques.
- Les chiffres de rappel du « rappel de la présélection » (combien de vraies paires survivent au tri par figures) : jamais mesuré dans ces travaux ; à mesurer sur notre vérité terrain avant de s'engager.

Risques pour notre corpus : échec de la détection de pose sur les estampes tardives, schématiques ou de style non figuratif (Jenicek) ; copies partielles ou recadrées (une seule figure sur trois conservée) ; miroir (gérer par minimum des deux distances) ; figures en bonne partie occultées.

Recommandation d'expérience minimale : mesurer, sur nos paires de copies connues, (i) rappel@50 d'une liste restreinte par pose (Jenicek-like, miroir inclus) et (ii) rappel@50 par embedding global ; puis de leur union. Si l'union gagne nettement, la pose sert de second canal de présélection plutôt que de remplaçant.

## Ce que je n'ai pas pu vérifier
- Chiffres de Johnson 2015 (403) ; tableaux détaillés d'AMES (PDF trop gros, HTML en échec) ; SemArt-HOI, CompoVista, ArtHOI (aucun papier trouvé) ; relations dans DEArt ; SuperGlue/LightGlue en réordonnancement ; PE-Spatial ; temps par paire de SegSwap ; chiffres 2021-2025 des travaux de graphes autres que PRISm et SPAN.
- Les résumés d'outil de lecture (pages web converties) peuvent contenir des erreurs de transcription ; les chiffres de Jenicek, Schroeder et Ufer viennent du texte brut du PDF, donc plus sûrs.
