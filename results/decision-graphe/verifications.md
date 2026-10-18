# Vérifications (tâche 3) — décision graphe de scène

Date : 17/10/2026. Règle : une URL par fait ; « non vérifié » quand je n'ai pas pu le lire. Aucun calcul lourd, aucune adresse e-mail dans les requêtes. Pour les sources brutes : `src/` (PDF du mémoire et texte extrait) et `raw/` restent sur le disque, hors git.

## a) CompoVista

- **Article** : *CompoVista: A Composition-Graph-Based Visual Analytics System for Compositional Analysis of Traditional Chinese Paintings*, arXiv:2607.07105 (cs.HC). https://arxiv.org/abs/2607.07105
- **Auteurs** : Dekun Qian, Ruiqi Yu, Li Ye, Yize Li, Fengling Zheng, Weigui Zheng, Yigang Wang, Jinchang Li, Zhiguang Zhou (même page).
- **Dates** : v1 soumise le 8 juillet 2026 ; v2 révisée le 29 juillet 2026 (même page).
- **Ce qu'il fait** : un système de visualisation analytique. Il s'appuie sur une représentation « CompoGraph » en quatre couches (entités, relations, vides, contexte) et permet à des historiens de l'art de définir des cohortes de peintures, de comparer des motifs de composition et de remonter aux peintures individuelles. Évaluation : étude de tâches avec 12 participants du domaine, deux études de cas et des entretiens d'experts (résumé de la page arXiv et des agrégateurs ; je n'ai pas lu le PDF complet).
- **Graphe appris ou annoté à la main ?** **Annoté à la main** : « Entity and relation annotations were manually assigned using predefined taxonomies » ; les métadonnées sont recopiées, seuls les vides (« voids ») sont proposés par traitement d'image puis corrigés par des annotateurs (32 % des propositions ont demandé une intervention). Aucun détecteur, modèle de relations ni VLM n'est décrit pour les entités et relations. Source : https://arxiv.org/html/2607.07105 (lecture des 100 000 premiers caractères).
- **Nombre d'images** : **960 peintures**, 7 587 instances d'entités, 6 180 instances de relations, 589 vides annotés. Corpus source : Chinese Ancient Open Dataset (CAOD), cité par l'article (même source).
- **Relations annotées** : **37 types** en deux familles. Localisation (au-dessus, derrière, le long de, à l'intérieur) et événement (tenir, être assis sur, regarder). Six classes d'entités (humain, animal, artefact, végétation, relief, eau), 109 étiquettes fines.
- **Code, données, poids publics ?** La page arXiv ne mentionne **aucune publication de code, de données ou de poids**. Seuls le PDF, la version HTML et les sources TeX sont proposés. Licence de l'article : CC BY-NC-ND 4.0, qui couvre l'article et non un logiciel ou des données. https://arxiv.org/abs/2607.07105
- Article compagnon cité : VisTCP (arXiv 2607.05841), graphe de connaissances pour la peinture chinoise traditionnelle, mêmes auteurs en partie : https://papers.cool/arxiv/2607.05841 (non ouvert).

## b) Licence de RelTR

- Le dépôt https://github.com/yrcong/RelTR n'a **pas de fichier LICENSE** à la racine de la branche principale : la requête `github.com/yrcong/RelTR/blob/main/LICENSE` renvoie **HTTP 404**.
- La page principale du dépôt ne montre **aucune licence** dans la section « About » ni dans la liste des fichiers (lecture du 17/10 : https://github.com/yrcong/RelTR).
- Le README ne mentionne pas de licence (lecture de la page principale, qui affiche le README : même source).
- Conclusion : **rien**. Licence des poids Google Drive : **non vérifié**. À demander aux auteurs avant tout usage hors recherche.

## c) SemArt-HOI (mémoire de Maya Antoun, AUB)

- **Mémoire** : *Human Object Interaction Detection in Paintings using Multi-Task Learning*, Maya Antoun, AUB, soutenu le 20 juin 2023 (date de la page : 17 juillet 2023). Page : https://scholarworks.aub.edu.lb/items/85ff8f94-a397-4c9b-b13d-18ecc52d30d7 ; handle http://hdl.handle.net/10938/24097 (mentionné dans la page).
- **PDF** : ouvert (21,4 Mo, téléchargé dans `src/aub-semart-hoi.pdf`, texte extrait dans `src/aub-semart-hoi.txt`). Source : https://scholarworks.aub.edu.lb/bitstreams/11a7d9f5-0722-4a3d-8625-4f2b65a7c0b1/download
- **Chiffres du mémoire (section 5.2, tableau 5.4, p. 68-69)** :
  - **14 187 images** étiquetées (sur 21 384 de SemArt ; seules celles avec une interaction humain–objet sont gardées).
  - **99 classes d'objets**, **66 verbes**, **248 interactions** (couples verbe–objet).
  - **73 683 boîtes** d'objets et **40 101 étiquettes d'interaction**.
  - Annotation : **à la main par l'auteur**, avec l'outil Make Sense, en deux étapes (boîtes, puis flèche entre les deux entités + verbe, p. 67-68).
- **Téléchargeable ?** Le mémoire **ne dit pas** que le jeu est publié, et je n'ai trouvé ni URL de dépôt ni licence. **non vérifié** ; il faudrait écrire à l'auteur.
- **Base** : SemArt (Garcia & Vogiatzis), 21 384 images, licence CC BY-NC 4.0, ~3 Go, usage non commercial : https://researchdata.aston.ac.uk/id/eprint/380/ (page lue ; annotations : commentaires artistiques, pas de relations entre figures).

## d) Autres jeux de données (recherche de 30 min)

Les jeux ci-dessous ont des relations ou des interactions entre figures, ou une annotation de relations, mais aucun ne fournit de graphe de scène publié et utilisable avec licence claire.

1. **Graphe de connaissances de peinture chinoise traditionnelle (IEEE DataPort)** — 649 peintures (JPG), 2,64 Go en zip, DOI 10.21227/63pm-1f21. Entités : humain, paysage, animal, artefact. **Relations : événement et lieu** (pas d'interaction fine). Annotation : **non vérifié** (manuelle ou automatique). **Accès par abonnement IEEE DataPort** ; licence **non vérifié**. Créé le 23/10/2025, auteur unique Fengling Zheng. https://ieee-dataport.org/documents/knowledge-graph-based-representation-traditional-chinese-painting
2. **DEArt** (ECCV workshops 2022, Reshetnikov et al., arXiv 2211.01226) — plus de 15 000 images, 69 classes, 12 poses ; **boîtes et poses, pas de relations** dans la version de base. Archive publique B2SHARE (annots_pub.zip ~18,2 Go) : https://b2share.eudat.eu/records/k23em-7tn08. Licence de la version originale : **non vérifié**. Une extension (dépôt GitLab ECCV AI4DH 2024) indique CC BY-NC-SA pour la version étendue, avec génération de relations sans annotation manuelle dans un article de 2025 (Journal of Cultural Heritage, non ouvert). Sources : https://arxiv.org/pdf/2211.01226 ; https://gitlab.liris.cnrs.fr/anr-aaa/eccv-ai4dh-2024/-/blob/67ad7e2b9c42c50793197763efc32b556a5bccf1/README.md
3. **FRAME** (Schneider, Göldl, Stalter, Vollmer, arXiv 2602.19133) — **texte**, pas image : 37 types d'entités, relations typées, annotation manuelle de descriptions d'œuvres issues de catalogues de musées. Taille : non vérifié. Licence : non vérifié (seule la licence arXiv). Pas d'URL de téléchargement. https://arxiv.org/abs/2602.19133
4. **Vases attiques, narrations** (Madhu, Kosti, Bendschus, Reinhardt, Maier, CAA 2020) — « plusieurs milliers » d'images de vases des VIe–Ve siècles av. J.-C. ; annotations d'attributs, de gestes et de poses, et de thèmes de scène (enlèvement, poursuite). Relations entre figures : non vérifié. Données et licence : non vérifié (seulement un lien BibTeX). https://cris.fau.de/publications/234588860?lang=en_GB
5. **Kerameikos.org** : données liées (ODbL) et métadonnées de vases de musées (Getty 3 655, British Museum 4 244, Met 15 722 — licences variables, CC0 pour le Met). **Aucune annotation de relation** : hors sujet, mais utile pour les images et métadonnées. https://www.kerameikos.org/datasets
6. **Berlin SBB-PK, motifs annotés** (Zenodo, 2023) : régions d'images avec ICONCLASS, pas de couche de relations (vu dans un résultat de recherche seulement, non ouvert) : https://zenodo.org/records/8358608 — **non vérifié**.

Aucun résultat sur VISART (ECCV) ni sur les actes CHR/DH avec des relations entre figures annotées : la recherche dans les actes de VISART ne donne que le People-Art (boîtes et points-clés, pas de relations). https://arthist.net/archive/41794

## Constats principaux

1. **CompoVista** : article arXiv du 8/07/2026 (v2 le 29/07). Graphe de composition de 960 peintures, annoté à la main (37 types de relations, dont événements « tenir » et « regarder »), mais pas de modèle de génération. Ni code ni données annoncés : il faudrait demander les annotations aux auteurs.
2. **RelTR** : aucune licence trouvée (LICENSE absent, 404). À demander aux auteurs.
3. **SemArt-HOI** : 14 187 images, 99 objets, 66 verbes, 248 interactions, annoté à la main par l'auteur ; **non publié à ma connaissance**, sans URL ni licence trouvées. Base SemArt en CC BY-NC 4.0.
4. **Autres jeux** : aucun graphe de scène publié avec licence claire. Les candidats les plus proches sont SemArt-HOI (non publié), le jeu de peinture chinoise (accès payant, relations seulement événement/lieu) et DEArt (relations ajoutées par une extension non publiée à ma connaissance).
