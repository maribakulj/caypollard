# Précédents directs : SGG / VRD / HOI appliqués aux œuvres d'art (état au 17/10/2026, PARTIEL)

Avertissement : la recherche a été interrompue par une limite de session API (~20 appels). Couverture incomplète. Tout ce qui n'a pas été ouvert sur une page primaire est marqué « non vérifié ».

## Quels articles appliquent SGG / VRD / HOI à l'art ?

### Takeaway
Un seul précédent HOI dédié aux peintures a été trouvé et vérifié : la thèse de master de Maya Antoun (AUB, juillet 2023), qui introduit SemArt-HOI. Aucun entraînement de SGG/VRD « général » dédié à l'histoire de l'art n'a été trouvé (absence de résultat ≠ preuve d'absence, recherche tronquée). Les travaux voisins (composition, pose/regard, graphes de connaissances de peinture chinoise, VQA) ne produisent pas de triplets sujet-prédicat-objet évalués.

### Cited Findings
- SemArt-HOI : thèse de master « Human Object Interaction Detection in Paintings using Multi-Task Learning », Maya Antoun, American University of Beirut, 17/07/2023 (dir. Daniel Asmar d'après un résultat de recherche). Méthode : Faster R-CNN pour personnes/objets, traits visuels + spatiaux, graphe de connaissances (GCN), 4 tâches de classification auxiliaires. Résultats annoncés : +1,19 % (mono-tâche) et +1,51 % (multi-tâche) sur un modèle HOI transformeur one-stage de l'état de l'art ; entraînement 4 fois plus rapide. Licence non indiquée sur la page ; PDF de 20,4 Mo (AntounMaya_2023.pdf), handle http://hdl.handle.net/10938/24097. — [AUB ScholarWorks](https://scholarworks.aub.edu.lb/items/85ff8f94-a397-4c9b-b13d-18ecc52d30d7)
- VÉRIFIÉ dans le PDF de la thèse (AntounMaya_2023.pdf, section 5.2, Table 5.4 et Table 5.7) : SemArt-HOI = 14 187 images (sur 21 384 SemArt, seules celles avec paires en interaction), 99 classes d'objets, 66 verbes, 248 interactions distinctes, 73 683 boîtes d'objets, 40 101 labels d'interaction ; annotation par l'auteure seule avec Make Sense (boîtes + flèche + verbe, triplet <humain, prédicat, objet>). Résultats SemArt-HOI test (%mAP) : HOI-Paint 18,32 ; HOI-Paint-MTL 18,64 ; QAHOI 17,13 (entraîné 8 jours sur 3 V100 contre 1 j 18 h sur 1 V100). Donc les +1,19 / +1,51 sont des points de mAP absolus, et le mAP absolu reste très bas (~18 %). Limites notées par l'auteure : pas de label « no interaction », un seul verbe par paire ; annotation mono-annotateur (aucun accord inter-annotateurs mentionné dans les passages lus). Jeu SemArt de base : researchdata.aston.ac.uk/id/eprint/380 ; disponibilité publique de SemArt-HOI NON établie. — [PDF AUB](https://scholarworks.aub.edu.lb/bitstream/handle/10938/24097/AntounMaya_2023.pdf)
- SemArt-HOI = extension de SemArt (21 384 échantillons) avec annotations d'instances et classes d'interaction. — [AUB](https://scholarworks.aub.edu.lb/items/85ff8f94-a397-4c9b-b13d-18ecc52d30d7) ; [SemArt](https://noagarcia.github.io/SemArt/)
- Les chiffres 14 187 images / 66 verbes sont confirmés par le PDF (voir ci-dessus).
- Une version article existe probablement : ScienceDirect S2212054824000493 (« Human object interaction detection in paintings using multi-task learning », préfixe S2212054 = a priori Digital Applications in Archaeology and Cultural Heritage, 2024) — page renvoyant 403, revue/année NON VÉRIFIÉES. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2212054824000493)
- CompoVista (arXiv 2607.07105, cs.HC, v1 08/07/2026, v2 29/07/2026), Qian et al. : système d'analytique visuelle ; « CompoGraph » modélise la composition en 4 couches (entités, relations, vides, contexte) pour peintures chinoises traditionnelles ; études de cas, étude utilisateur, entretiens d'experts. Pas de chiffres ni de jeu de données nommé sur la page abstract. Ce n'est pas un SGG appris évalué. — [arXiv](https://arxiv.org/abs/2607.07105)
- VisTCP (arXiv 2607.05841, cs.HC, 07/07/2026), Zhou et al. : cadre de visualisation human-in-the-loop pour construire une représentation en graphe de connaissances de peintures chinoises ; modèle entraîné sur annotations d'experts pour extraire objets et relations ; l'abstract souligne que les méthodes d'image standard échouent car objets/événements diffèrent des images naturelles. Aucun chiffre ni jeu de données nommé sur la page. — [arXiv](https://arxiv.org/abs/2607.05841)
- Madhu et al., « Understanding Compositional Structures in Art Historical Images using Pose and Gaze Priors », ECCV 2020 workshops (VISART V), arXiv 08/09/2020 : détecte « action regions » et « action lines » via pose et regard, sans entraînement supplémentaire ; étude utilisateur (experts/non-experts) corrélée à la méthode ; code sur GitHub. Proxy de relations (lignes d'action entre figures), pas un SGG. — [arXiv](https://arxiv.org/abs/2009.03807)
- Madhu, Kosti, Bendschus, Reinhardt, Maier, « Computer Vision Understanding of Narrative Strategies on Greek Vases », CAA 2020 : schémas d'interaction humaine (conduite de la mariée, enlèvement, poursuite) sur plusieurs milliers d'images de vases attiques ; attributs, gestes, poses dans un espace joint ; résultats préliminaires, pas de métrique sur la page. — [FAU CRIS](https://cris.fau.de/publications/234588860)
- Pages ETRI (Corée) sur un « visual narrative » de peintures par scene graph et Mobile Visual Search : la requête a renvoyé les ids 61226 et 62978 ; la page 61226 était inaccessible (ECONNREFUSED), 62978 non ouverte. Contenu NON VÉRIFIÉ, hormis les extraits du moteur de recherche. — [ksp.etri.re.kr 61226](https://ksp.etri.re.kr/ksp/article/read?id=61226)
- Relations dans la bande dessinée (domaine non photo analogue) : Chen, « Hierarchical Knowledge Graphs for Story Understanding in Visual Narratives », arXiv 2506.10008 (v1 14/04/2025, v2 15/11/2025, ICIDS 2025) : graphes panneau/événement/macro-événement (personnages, objets, actions, dialogues) sur un sous-ensemble annoté manuellement de Manga109 ; pas de résultats quantitatifs (cadre symbolique). — [arXiv](https://arxiv.org/abs/2506.10008)
- Manga109Dialog (arXiv 2306.17469) : 132 692 paires locuteur-texte ; les auteurs utilisent une génération de scene graph pour lier texte et personnage (d'après résultat de recherche ; page non ouverte → NON VÉRIFIÉ en détail). — [arXiv](https://arxiv.org/abs/2306.17469)
- VQArt-Bench (arXiv 2510.12750, oct. 2025) : benchmark VQA d'art évaluant 14 MLLM ; l'abstract cite « complex visual relationships » ; des dimensions « spatial relations » et « instance interaction » et le chiffre de 14 463 questions viennent d'un résumé de recherche, pas de la page ouverte (qui ne donne ni taille ni licence) → NON VÉRIFIÉ. — [arXiv](https://arxiv.org/abs/2510.12750)
- Enquête « Open-vocabulary models for object detection and segmentation in visual art: survey and comparative study » (Multimedia Tools and Applications, DOI 10.1007/s11042-026-21443-y, 2026) : d'après la recherche, OWLv2 meilleur ; cite SemArt-HOI. Page inaccessible (redirection d'authentification) → NON VÉRIFIÉ. — [Springer](https://link.springer.com/article/10.1007/s11042-026-21443-y)
- Pistes de recherche à relancer (non traitées : limite de session) : VISART ECCV/ICCV (autres éditions), CHR, DSH, JOCCH, IJDAR, HAL, ArtGraph, Iconclass AI test set, Emblematica, HORAE, Mandragore, ArtPedia, Brueghel/ArtMiner, MET, PeopleArt, ArtDL.

### Inferences
- Le champ « HOI/SGG pour l'art » est quasi vide : un seul benchmark dédié (SemArt-HOI) et il est le fait d'une thèse, sans trace d'adoption vérifiée.
- Les travaux 2026 sur peintures chinoises (CompoVista, VisTCP) relèvent de l'analytique visuelle avec experts ; ils confirment que les modèles d'images naturelles ne conviennent pas (VisTCP) mais ne fournissent pas de métriques SGG comparables.
- Le plus proche d'un « SGG pour l'histoire de l'art » : SemArt-HOI (HOI), VisTCP (extraction objets+relations, domaine chinois) ; aucun modèle SGG général entraîné sur art européen n'a été trouvé.

### Gaps
- PDF Antoun : licence et disponibilité publique du jeu, ainsi que l'analyse d'erreurs par genre, non lues.
- Revue/année exactes de la version article SemArt-HOI.
- Existence éventuelle d'autres papiers VISART/CHR/DSH/JOCCH/IJDAR : non recherchés.
- ETRI : contenu non lu.

## Jeux de données avec relations ou dérivables

### Takeaway
Aucun jeu vérifié ne fournit de triplets sujet-prédicat-objet pour l'art en dehors de SemArt-HOI (dont la disponibilité est non vérifiée). Plusieurs jeux fournissent boîtes + poses (DEArt, PoPArt, Human-Art) dont des relations pourraient être dérivées.

### Cited Findings
- DEArt (Reshetnikov, Marinescu, More Lopez), arXiv 2211.01226, ECCV 2022 VISART VI : >15 000 images (~80 % non iconiques), boîtes pour 69 classes (>50 spécifiques au patrimoine : êtres imaginaires, entités symboliques), 12 poses pour objets humanoïdes ; peintures env. XIIe-XVIIIe s. ; article en CC BY 4.0, pas de lien de téléchargement ni de licence du jeu sur la page arXiv. — [arXiv](https://arxiv.org/abs/2211.01226)
- Extension DEArt, Journal of Cultural Heritage 2025 (DOI 10.1016/j.culher.2025.07.022, PII S1296207425001542) : d'après un résultat de recherche (dépôt URV), ajoute une génération automatique de labels objets et relations, évaluée par utilisateurs. Page éditeur non ouverte (redirection) → NON VÉRIFIÉ. — [DOI](https://doi.org/10.1016/j.culher.2025.07.022)
- RichArt (HF MihaiBogdanBindila/RichArt) : 7 087 peintures, 20 882 annotations de région, 6 233 labels uniques, splits 5 669/709/709, 1,81 Go ; boîtes [x_min,y_min,x_max,y_max] avec « symbolic narratives » (descriptions ~35 mots) et score GroundingDINO ; médiane 2 objets/image (max 8) ; sources Web Gallery of Art 76,5 %, Met 12,9 %, WikiArt 10,6 % ; licence cc-by-nc-4.0 ; aucune annotation de relation décrite. — [HF card](https://huggingface.co/datasets/MihaiBogdanBindila/RichArt)
- Human-Art (Ju et al., CVPR 2023, arXiv 2303.02760) : ~50k images, >123k instances de personnes, 5 scénarios naturels + 15 artificiels ; boîtes, keypoints 2D, points de contact (self-contact), texte ; code github.com/IDEA-Research/HumanArt (d'après résultat de recherche ; licence non vérifiée). — [CVPR page](https://cvpr.thecvf.com/virtual/2023/poster/22495)
- PoPArt (Schneider, Vollmer, J. Comput. Cult. Herit., 2024, DOI 10.1145/3696455) : figures et keypoints en art historique ; détails non ouverts → NON VÉRIFIÉ. — [ACM](https://dl.acm.org/doi/10.1145/3696455)
- Springstein et al., ACM MM 2022 : pose semi-supervisée en images d'histoire de l'art, jeu avec boîtes + keypoints ; comble le domain gap art/photo (résumé de recherche) → NON VÉRIFIÉ. — [ACM](https://dl.acm.org/doi/10.1145/3503161.3548371)
- IconArt (Zenodo 4737435, ECCV 2018 VISART) : 5 955 images WikiCommons (train 2 978 / test 2 977), boîtes pour 7 classes, pas de relations (résumé de recherche, page non ouverte) → NON VÉRIFIÉ. — [Zenodo](https://zenodo.org/records/4737435)
- Manga109Dialog, FGC/ICDAR : boîtes de cases/personnages et liens bulle-personnage, sans relations personnage-personnage étiquetées (résumé de recherche, non ouvert). — [Manga109Dialog](https://arxiv.org/abs/2306.17469)

### Inferences
- RichArt est sous licence NC et sans relations : utile comme source d'images + descriptions de régions, pas comme vérité terrain relationnelle.
- DEArt (boîtes + poses) et Human-Art/PoPArt (poses) sont les meilleurs candidats pour dériver des relations géométriques, au prix d'un proxy non validé.

### Gaps
- Non examinés : ArtDL, PeopleArt, SemArt (licence), Iconclass AI test set, ArtPedia, Brueghel/ArtMiner, MET Open Access, Emblematica Online, HORAE, Mandragore.
- Licences/téléchargement de DEArt, PoPArt, SemArt-HOI non établis.

## Résultats chiffrés et domain gap photo-art

### Takeaway
Peu de chiffres vérifiés. SemArt-HOI : gains relatifs de +1,19 / +1,51 (unité non précisée, probablement points de mAP) sur un détecteur HOI transformeur ; VisTCP affirme que les méthodes d'images naturelles échouent sur les peintures chinoises ; Human-Art affirme que les modèles échouent sur les humains « artificiels ».

### Cited Findings
- Gains SemArt-HOI : voir plus haut. — [AUB](https://scholarworks.aub.edu.lb/items/85ff8f94-a397-4c9b-b13d-18ecc52d30d7)
- VisTCP : méthodes standard médiocres car objets et événements de peintures chinoises diffèrent des images modernes. — [arXiv](https://arxiv.org/abs/2607.05841)
- Human-Art : la plupart des travaux centrés humain portent sur des photos ; les modèles échouent sur humains artificiels. — [CVPR](https://cvpr.thecvf.com/virtual/2023/poster/22495)
- DEArt : les modèles courants performent mal sur les œuvres ; le transfert par fine-tuning atteint une précision comparable à l'état de l'art sur images génériques. — [arXiv](https://arxiv.org/abs/2211.01226)

### Inferences
- Aucun mode d'échec propre aux relations (par opposition à la détection) n'a pu être documenté.

### Gaps
- Chiffres absolus (mAP HOI, Recall@K SGG) et modes d'échec détaillés : à extraire du PDF Antoun et des articles DEArt 2025.

## Existe-t-il un SGG entraîné spécifiquement pour l'histoire de l'art ? Sinon, le plus proche ?

### Takeaway
Non trouvé (recherche partielle). Plus proche : SemArt-HOI/Antoun (HOI sur peintures européennes, SemArt) ; DEArt 2025 (labels d'objets et relations générés automatiquement, à vérifier) ; VisTCP (peinture chinoise, human-in-the-loop).

### Cited Findings
- Recherches de « scene graph generation artworks/paintings » n'ont renvoyé que des travaux sur images naturelles (Visual Genome, VRD) ; pas de SGG art spécifique. — [Survey SGG](https://arxiv.org/abs/2201.00443)
- Dans l'analogue manuscrits enluminés, les travaux trouvés détectent des illustrations, pas des relations (arXiv 2601.05269, 2026). — [arXiv](https://arxiv.org/abs/2601.05269)

### Inferences
- Opportunité : un SGG art-historique dérivé de boîtes+poses+légendes reste à construire.

### Gaps
- Recherche non terminée (limite de session) ; à relancer sur VISART 2024-2026, CHR, ACL Anthology, HAL, GitHub.
