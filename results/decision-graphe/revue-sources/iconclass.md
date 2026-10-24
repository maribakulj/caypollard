# Revue : modèles qui attribuent des codes Iconclass à des images, et leur disponibilité

Date de la revue : 25/10/2026. Tout ce qui est écrit ici vient d'une page ou d'un PDF effectivement ouvert (adresse donnée). Ce que je n'ai pas pu vérifier est signalé « NON VÉRIFIÉ ». Aucun poids téléchargé, aucun calcul lourd.

Note de méthode : le serveur de la CVF (openaccess.thecvf.com) refuse l'outil de lecture web (403) ; j'ai donc téléchargé le PDF du papier WACV avec curl et lu le texte (pdftotext). Pour la communication Posthumus 2022, le PDF Zenodo est une présentation de diapositives (OCR), pas un article avec tableaux.

## Résumé de décision

- Le seul « brique » sérieuse, gelée, sortant des codes Iconclass par image sur tout le vocabulaire (≈20 000 codes) est le modèle CAT de Springstein et al. (WACV 2024). Code, poids (MIT, 4,1 Go) et données (CC BY 4.0) existent et sont en ligne depuis mars 2026. Mais ses scores sont faibles (mAP 0,17 sur le jeu de test Iconclass AI, 0,37 sur les codes bien représentés) et le dépôt de code n'affiche aucune licence. Il prédit au niveau image, pas au niveau région.
- Rien de ce que j'ai trouvé ne prédit des codes Iconclass par région / boîte englobante. Les jeux avec boîtes (IconArt, DEArt) utilisent des noms de classes, pas des codes Iconclass.
- Les alternatives sont : (a) recherche par similarité CLIP vers des images déjà indexées (FIZ Karlsruhe / Arkyves), utilisable seulement via leur démo ; (b) VLM génériques en zero-shot sur peu de classes (Gemini 2.5 Pro, 83 % sur 10 saints Iconclass) ; (c) LLM + RAG sur des descriptions (Thomas, 87-92 % de précision, mais sur deux branches seulement et à partir de pages entières) ; (d) CARIS (YOLOv8 + règles), preuve de concept sans chiffres et sans licence.

## 1. Springstein et al., WACV 2024 (« Visual Narratives »)

Sources ouvertes : PDF https://openaccess.thecvf.com/content/WACV2024/papers/Springstein_Visual_Narratives_Large-Scale_Hierarchical_Classification_of_Art-Historical_Images_WACV_2024_paper.pdf (via curl) ; README GitHub https://github.com/TIBHannover/iconclass-classification (via l'API GitHub /readme) ; carte Hugging Face https://huggingface.co/springsteinm/iconclass_cat ; Zenodo https://zenodo.org/records/18980670.

Ce qu'il prédit : des codes Iconclass, au niveau de l'image entière, multi-étiquettes, hiérarchiques. Vocabulaire : 20 596 codes distincts dans ICARUS (le papier dit « plus de 20 000 concepts »), 12 488 dans ICAI.

Données (PDF, section 4) :
- ICAI = Iconclass AI Test Set : 87 744 images, 362 561 annotations, 12 488 codes distincts, échantillonnées dans la base Arkyves. Oui, il est inclus : il sert de jeu de test (Tableau 2 : « ICAI and ICARUS test sets »).
- ICARUS (nouveau) : 477 569 images, 1 328 417 annotations, 20 596 codes distincts, 19 collections publiques ; découpage 80/10/10.
- NON VÉRIFIÉ : si ICAI sert aussi à l'entraînement (le papier renvoie le détail et l'unification des annotations au supplément, que je n'ai pas ouvert ; le dépôt Hugging Face contient des fichiers « brill/ train / test / val .jsonl », ce qui suggère un usage de ICAI aussi en entraînement, à confirmer).

Architecture (PDF, sections 3 et 5.1) : en deux étapes. (1) Pré-entraînement contrastif d'un CLIP ViT-B/16 sur ICARUS avec des descriptions synthétiques fabriquées à partir des mots-clés Iconclass (trois stratégies : mots-clés, BLIP, GPT ; BLIP retenu). (2) Décodeur hiérarchique « CAT » (cross-attention transformer, 3 couches de décodeur, 8 têtes) sur l'encodeur image gelé de CLIP ; il prédit niveau par niveau de la hiérarchie ; 5 codes max par image à l'entraînement, 30 itérations d'inférence.

Métriques : AP moyenne (mAP) sur les codes ayant au moins un exemple dans le test, avec seuils sur le nombre d'images d'entraînement par code (Tableau 2, BLIP) :

| Jeu de test | Classifieur | >0 | >10 | >100 | >1000 |
|---|---|---|---|---|---|
| ICAI | CAT | 0,1715 | 0,1803 | 0,2012 | 0,2737 |
| ICAI | Flat-H (meilleure base) | 0,0638 | 0,0777 | 0,1148 | 0,2107 |
| ICARUS | CAT | 0,1714 | 0,1889 | 0,2407 | 0,3716 |
| ICARUS | Flat-H | 0,0789 | 0,0946 | 0,1507 | 0,2935 |

Lecture : même sur les codes très fréquents (>1000 exemples), mAP 0,27 (ICAI) à 0,37 (ICARUS). Le CLIP seul en zero-shot est quasi nul (0,003). Le papier juge les résultats « prometteurs » et voit surtout un usage semi-automatique (suggestion puis validation par un expert). Une évaluation qualitative avec des experts est mentionnée, détails dans le supplément (NON VÉRIFIÉ).

Résultats par branche Iconclass (31 figure humaine, 33 relations, objets) : le papier principal n'en donne pas. NON VÉRIFIÉ (le supplément n'est pas ouvert). On peut les recalculer soi-même avec les poids et le jeu de test.

Disponibilité (vérifiée) :
- Code : https://github.com/TIBHannover/iconclass-classification existe (API GitHub : dernier push 17/03/2026, créé 02/11/2023). Dossiers encoders, decoders, heads, models ; scripts train.py, infer.py, test.py, export.py ; requirements.txt et singularity.def. Licence : aucune détectée (l'API GitHub renvoie license = null ; la page web n'affiche pas de LICENSE). Donc par défaut tous droits réservés côté code : à demander aux auteurs.
- Poids : https://huggingface.co/springsteinm/iconclass_cat, licence MIT (étiquette de la carte). Fichiers listés par l'API HF : cat_icarus_40000.ckpt (4 122 950 205 octets ≈ 4,1 Go), icarus_classifiers.jsonl, icarus_mapping.jsonl, icarus_ontology.jsonl, plus des jsonl brill/ et train_blib.jsonl. La carte modèle est vide (README de 21 octets), sans mode d'emploi ni métriques. NON VÉRIFIÉ : ce que contient exactement le .ckpt (CLIP + CAT ensemble ou non), je n'ai pas téléchargé.
- Données : Zenodo « ICARUS » v1.0, publié le 12/03/2026, CC BY 4.0, 117,8 Go (data.jsonl 629 Mo ; images en archive multi-parties). « Images are included only when they are in the public domain or otherwise freely accessible » : le jeu n'embarque donc que les images libres, et la licence CC BY couvre les métadonnées/annotations, pas forcément chaque image (droits de chaque collection : NON VÉRIFIÉ).
- Calcul : le papier indique 40 000 itérations, batch 256, AdamW, lr 1e-4, pour chacune des deux étapes. Le type de GPU n'est pas donné dans le texte principal (NON VÉRIFIÉ, peut-être dans le supplément). Pour l'inférence seule : un ViT-B/16 + décodeur à 3 couches répété jusqu'à 30 fois par image ; modeste, mais je ne l'ai pas mesuré.

Peut-on le brancher comme brique gelée « image vers codes Iconclass » ? Oui, c'est le candidat le plus direct : poids MIT, sortie hiérarchique (on peut tronquer à une profondeur sûre). Réserves : (1) précision basse, à traiter comme suggestion à filtrer, pas comme vérité ; (2) licence du code absente ; (3) l'instruction d'usage est à reconstituer depuis infer.py (NON VÉRIFIÉ que cela tourne sans accroc, je ne l'ai pas lancé) ; (4) entraîné sur images de collections, pas de gravures de presse ou d'emblèmes à la résolution de notre corpus (NON VÉRIFIÉ pour le transfert).

## 2. Banar, Daelemans, Kestemont, ICAART 2021

Source ouverte : PDF https://www.scitepress.org/Papers/2021/103906/103906.pdf (page https://www.scitepress.org/Papers/2021/103906 vue en recherche).

- Prédit : par récupération croisée (retrieval), l'image (ou le titre) est appariée aux définitions anglaises des codes Iconclass ; niveau image entière ; vocabulaire : codes jusqu'à la profondeur 5, soit 10 418 codes candidats ; la hiérarchie n'est pas exploitée.
- Données : 26 725 objets du RKD (Pays-Bas), avec codes Iconclass manuels, titres néerlandais et anglais, photographie ; 2 000 pour le développement, 2 000 pour le test, 22 725 pour l'entraînement.
- Modèle : SAEM (Faster R-CNN ResNet-101 pré-entraîné sur Visual Genome, gelé, pour les régions ; BERT multilingue gelé pour les titres) ; entraîné 30 époques sur un seul GTX 1080 Ti (11 Go).
- Scores (Recall@1 / @5 / @10), Tableau 2 : image seule (cross-modal) 13,10 / 19,60 / 23,20 ; titre anglais seul 62,80 / 77,30 / 80,75 ; meilleur modèle multimodal (titres NL + EN + image) 70,05 / 80,35 / 83,10. Conclusion des auteurs : le texte domine, l'image apporte un complément ; l'image seule n'est « pas prometteuse ».
- Artefacts : aucun dépôt propre à l'article n'est cité ; seul le code SAEM d'autrui (https://github.com/yiling2018/saem) est référencé. Pas de poids. Le suivi (INSIGHT) : je n'ai trouvé aucune publication vérifiable portant ce nom ; NON VÉRIFIÉ. Le papier de Springstein cite aussi un travail Banar de 2023 sur la recherche multimodale (via la bibliographie de CARIS, https://arxiv.org/html/2602.19698v1), que je n'ai pas ouvert.
- Brique gelée ? Non : pas de poids, et surtout l'image seule donne R@1 = 13 %.

## 3. Posthumus, Brandhorst, Sack, VISART 2022 (« The Art Historian's Bicycle Becomes an E-Bike »)

Sources ouvertes : fiche Zenodo https://zenodo.org/records/7225417 ; PDF de 23 pages de ce dépôt (lu par OCR : c'est un diaporama).

- Principe (diapo « Mechanism ») : CLIP + index FAISS ; on cherche les images les plus proches dans une collection d'environ 750 000 objets déjà indexés en Iconclass, on récupère leurs notations, on trie par fréquence, on affiche les n meilleures. Résultat image entière.
- Licence du texte : CC BY 4.0. Publié le 23/10/2022 (v1 ; la fiche dit qu'une version plus récente existe, non ouverte).
- Chiffres d'évaluation : aucun dans le diaporama que j'ai lu. L'abstract Zenodo parle d'une évaluation du problème de classification multi-étiquette extrême ; les chiffres sont NON VÉRIFIÉS (peut-être dans la version plus récente).
- Code / poids : aucun lien (fiche Zenodo). 
- Brique gelée ? Pas en tant que tel ; voir la mise en œuvre de 2023 ci-dessous.

## 4. Santini et al. 2023, « Multimodal Search on Iconclass using Vision-Language Pre-Trained Models »

(Ce n'est pas un article de Banar ; auteurs : Santini, Posthumus, Tan, Bruns, Tietz, Sack.) Source ouverte : https://arxiv.org/pdf/2306.16529

- Prédit : à partir d'une image ou d'un texte, des codes Iconclass obtenus par k plus proches voisins parmi 531 172 images d'Arkyves (2 526 145 notations, 90 347 uniques) avec les embeddings CLIP d'OpenAI et un index FAISS de 2 Go ; image entière.
- Évaluation : seulement une enquête de préférence auprès de 10 volontaires (105 préférences pour la recherche multimodale contre 104 pour le TF-IDF ; justesse 64 contre 72 ; exhaustivité 30 contre 17). Les auteurs écrivent que la qualité doit encore être estimée sur une vérité terrain objective. Aucune mesure automatique.
- Code : https://github.com/ISE-FIZKarlsruhe/iconclass/tree/main/multimodal existe (app.py, Dockerfile, requirements.txt, README ; CLIP + FAISS ; point d'accès GET similarity_text). Licence : non affichée. Il faut fournir soi-même la base d'images annotées et l'index (le README renvoie au forum Iconclass). Pas de poids propres : CLIP d'OpenAI.
- Brique gelée ? Techniquement oui (CLIP gelé + base annotée), mais la base Arkyves n'est pas donnée ; on pourrait la reconstruire avec le jeu ICARUS ou le test set ICAI.

## 5. CARIS, arXiv 2602.19698 (2026)

Sources ouvertes : https://arxiv.org/abs/2602.19698 et https://arxiv.org/html/2602.19698v1 ; page GitLab https://gitlab.geist.re/pro/caris-iconclass.

- Titre : « Iconographic Classification and Content-Based Recommendation for Digitized Artworks » (Kutt et Baczyński ; CARIS = Classification and Recommendation for ICONCLASS System).
- Prédit : des codes Iconclass par correspondance de mots-clés entre les étiquettes d'objets détectés par YOLOv8 et les mots-clés Iconclass (bibliothèque iconclass), plus des règles JSON qui déduisent des sujets abstraits (cerf + chien + cheval + homme suggère la chasse). Les objets sont détectés par région (YOLO) mais la sortie est une liste de codes par image.
- Données d'entraînement de YOLOv8 : non précisées (le réglage fin sur des libellés alignés Iconclass est listé comme travail futur). Exemples sur 2 tableaux du domaine public. Le texte mentionne le jeu Iconclass AI (≈87 k images) comme inadapté à la reconnaissance visuelle parce que les images font 500 px au plus.
- Chiffres : aucune métrique quantitative, évaluation qualitative seulement. Exemples d'échec : chien pris pour un ours, faucon manqué ; l'union de mots-clés ambigus produit 1 231 codes contenant « human being ».
- Code : GitLab existe (1 commit, 133 Ko), « No license. All rights reserved ». Pas de poids.
- Brique gelée ? Non (preuve de concept, pas de licence, rappel d'objets limité).

## 6. Cetinic, « Iconographic Image Captioning for Artworks » (2021)

Source ouverte : https://arxiv.org/pdf/2102.03942 (lu en texte). La revue publiée est J. Imaging 2021 (« Towards Generating and Evaluating Iconographic Image Captions of Artworks », https://doi.org/10.3390/jimaging7080123, non ouvert).

- Ce qu'il fait : ne prédit pas de codes ; il génère une légende (liste de libellés Iconclass) par image, en affinant un modèle VLP (Oscar-type, pré-entraîné sur Conceptual Captions) sur 86 530 images du Iconclass AI Test Set, légendes obtenues en concaténant les définitions anglaises des codes (texte entre parenthèses supprimé).
- Scores sur son jeu de test : BLEU-1 14,8 ; BLEU-4 10,0 ; METEOR 11,7 ; ROUGE-L 31,9 ; CIDEr 172,1 (affinage 30 époques, lr 3e-5).
- Code, poids : aucun lien trouvé dans le texte lu (NON VÉRIFIÉ ailleurs). Retrouver les codes à partir des libellés générés exigerait une étape supplémentaire. Peu utile comme brique.

## 7. ArtDL (Milani et Fraternali) et études sur les saints

- ArtDL : fiche Zenodo https://zenodo.org/records/6473001 : 42 479 peintures, 19 classes de saints (Iconclass), licence CC BY 4.0 mais « available for academic research purposes only », les droits de chaque image restant à leurs propriétaires. Papier : https://arxiv.org/pdf/2010.11697 (non ouvert en entier ; la recherche en résume 71,2 % de précision, 70,9 % de rappel sur les saints : NON VÉRIFIÉ, chiffre tiré d'un résumé de moteur de recherche). Code annoncé : https://github.com/iFede94/ArtDL (non ouvert).
- Étude comparative 2025 : https://arxiv.org/html/2509.18839 (« Benchmarking Vision-Language and Multimodal Large Language Models... Christian Iconography ») : classification d'image entière sur 10 saints, trois jeux (ArtDL 1 864 images de test ; sous-ensemble Iconclass AI 863 images ; Wikidata 718). Précision de Gemini 2.5 Pro : ArtDL 90,45 %, Iconclass 83,31 % (84,82 % avec descriptions Iconclass), Wikidata 45,89 % ; meilleur modèle contrastif (SigLIP) : 72,21 % (ArtDL), 59,47 % (Iconclass), 66,29 % (Wikidata). Bases ResNet50 affinés : 84,44 %, 40,46 %, 43,97 %. Aucune analyse par région. Dépôt anonymisé (anonymous.4open.science) : non ouvert. Limites : 10 classes seulement, pollution possible des données d'entraînement des modèles fermés.

## 8. Autres travaux récents trouvés (2025-2026)

- Thomas, « Automating Iconclass: LLMs and RAG for Large-Scale Classification of Religious Woodcuts » : https://arxiv.org/pdf/2510.19986 (lu en texte). GPT-4o produit une description de la page entière (image + texte), puis une recherche vectorielle hybride la rapproche des codes Iconclass ; 87 % de précision à cinq niveaux et 92 % à quatre niveaux, limité aux branches 1 (Religion et magie) et 7 (Bible). Corpus ≈120 000 images (Ornamento) du Saint-Empire. Le code ou les données ne sont pas cités dans les passages lus (NON VÉRIFIÉ). Dépend d'une API fermée.
- Plus de modèle détectant des régions avec codes Iconclass : mes recherches n'en ont trouvé aucun. Les jeux de détection d'art (IconArt : ≈5 955 images, 7 à 10 classes ; DEArt : plus de 15 000 images, 69 classes) ont des boîtes mais des noms de classes libres, pas des codes Iconclass (résumé de recherche web ; pages https://zenodo.org/records/4737435 et https://merit.url.edu/es/publications/deart-dataset-ofeuropean-art/ non ouvertes en entier : NON VÉRIFIÉ). La conclusion d'ArtDL (arXiv 2010.11697) dit que des boîtes ou masques seraient nécessaires pour entraîner un détecteur.

## 9. Le Iconclass AI Test Set

Source ouverte : https://iconclass.org/testset/

- Taille : 87 749 images (la page) ; 87 744 chez Springstein et sur le miroir Hugging Face ; 86 530 images « valides » chez Cetinic. Zip de 3,1 Go ; data.json associe fichier et liste de notations.
- Résolution : 500 pixels au maximum sur le grand côté (confirmé par la page). Échantillon aléatoire d'Arkyves.
- Licences : codes Iconclass sous CC0 1.0. Les images : numérisées par Arkyves ou fournies par des institutions « sous licence ouverte », sans licence précisée image par image ; un fichier de droits est « prévu ». Demande d'indiquer l'usage et de partager les résultats. Une carte Hugging Face tierce (biglam/brill_iconclass) annoncerait CC0 : NON VÉRIFIÉ (vu seulement dans un extrait de recherche).
- Citation suggérée : Etienne Posthumus, « Iconclass AI Test Set », février 2020.
- Limite : 500 px et gravures/manuscrits ; les auteurs de CARIS le jugent inadapté à la reconnaissance visuelle fine.

## Tableau récapitulatif

| Travail | Niveau | Vocabulaire | Meilleur score cité | Poids / code | Licence vérifiée | Brique gelée ? |
|---|---|---|---|---|---|---|
| Springstein 2024 (CAT) | image | ≈20 600 codes, hiérarchique | mAP 0,17 (0,27 à 0,37 si >1000 ex.) | oui (HF 4,1 Go) + code GitHub | poids MIT ; code sans licence ; données CC BY 4.0 | oui, avec réserves |
| Banar 2021 | image | 10 418 codes (prof. 5) | image seule R@1 13 % ; multimodal 70 % | non | sans objet | non |
| Posthumus 2022 | image (kNN CLIP) | collection de ≈750 k objets | non chiffré dans les diapos | non | texte CC BY 4.0 | non |
| Santini 2023 | image (kNN CLIP) | 90 347 codes | enquête de préférence seulement | code d'appli (GitHub), base non fournie | non affichée | seulement si on reconstruit la base |
| CARIS 2026 | objets YOLO vers codes | mots-clés Iconclass | aucun chiffre | GitLab, tous droits réservés | non | non |
| Cetinic 2021 | légende | libellés | CIDEr 172 | non trouvé | NON VÉRIFIÉ | non |
| ArtDL / benchmark 2025 | image, 10-19 saints | 10-19 classes | Gemini 83-90 % | ArtDL données CC BY (usage académique) | oui pour ArtDL | non (trop étroit) |
| Thomas 2025 | page entière | branches 1 et 7 | précision 87-92 % | NON VÉRIFIÉ | NON VÉRIFIÉ | dépend de GPT-4o |

## Ce que je n'ai pas pu vérifier

- Le supplément du papier WACV (scores par branche, GPU, détail des 19 collections, évaluation par experts).
- Le contenu exact et le mode d'emploi du fichier .ckpt de 4,1 Go ; je n'ai rien lancé.
- Si ICAI figure dans l'entraînement de CAT.
- Les chiffres d'évaluation de la communication Posthumus 2022 (diaporama sans tableau ; une version plus récente existe).
- L'existence d'un suivi « INSIGHT » de Banar ; le travail Banar 2023 cité.
- Les licences par image des collections de ICARUS et du Iconclass AI Test Set.
- Les dépôts ArtDL (GitHub) et du benchmark 2025 (dépôt anonyme).
