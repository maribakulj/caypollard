# Iconographie computationnelle et VLM pour l'art (état au 17/10/2026)

Méthode : recherche web (WebSearch/WebFetch) le 17/10/2026. Les pages arXiv "abs" ont été lues ; les chiffres ne sont donnés que quand ils figurent dans une page lue (sinon "non vérifié"). Aucune adresse e-mail n'a été utilisée dans une requête.

## 1. Classification, légendage iconographique et benchmarks VLM/MLLM

### Takeaway
Sur la reconnaissance de saints (top-10 classes, étiquette unique), les MLLM propriétaires zero-shot (Gemini 2.5 Pro 90,45 % sur ArtDL) dépassent un ResNet50 fine-tuné (84,44 %), mais s'effondrent sur Wikidata (~45 %, au niveau du ResNet50) ; le few-shot n'aide pas. Aucun benchmark VLM dédié aux scènes narratives multi-figures/relations n'a été trouvé. "RichArt / MARGE-GD" n'a pas pu être retrouvé.

### Cited Findings
- ArtDL (Milani & Fraternali) : CNN sur 19 classes Iconclass, ~71,17 % précision, 70,89 % rappel, 70,25 % F1, 72,73 % AP ; le réseau s'appuie apparemment sur les attributs iconiques traditionnels ; jeu de 42 479 peintures (Zenodo). Chiffres issus d'un résumé de recherche web, non revérifiés sur le PDF — arXiv 2010.11697 (2020) : [arXiv](https://arxiv.org/abs/2010.11697v2) ; [Zenodo](https://zenodo.org/records/6473001)
- Spinaci, Klic, Colavizza, "Benchmarking VLMs and MLLMs in Zero-shot and Few-shot Scenarios: A study on Christian Iconography", arXiv 2509.18839, soumis 23/09/2025 (aussi publié Anthology of Computers and the Humanities vol. 3, 21/11/2025). 3 jeux (ArtDL, ICONCLASS, Wikidata), top-10 classes, 3 réglages : T1 étiquettes, T2 descriptions Iconclass, T3 few-shot (5 exemples) — [arXiv abs](https://arxiv.org/abs/2509.18839) ; [HTML avec tableaux](https://arxiv.org/html/2509.18839v1)
  - ArtDL (accuracy %) : Gemini 2.5 Pro 90,45 / 90,18 / 86,59 ; Gemini 2.5 Flash 88,20 / 87,02 / 84,71 ; GPT-4o 86,00 / 87,45 / 86,48 ; GPT-4o-mini 82,46 / 84,98 / 84,60 ; SigLIP so400m 53,86 / 70,55 / 56,38 ; CLIP ViT-L/14 30,58 / 44,31 / 31,71 ; ResNet50 fine-tuné 84,44.
  - ICONCLASS : Gemini 2.5 Pro 83,31 / 84,82 / 84,59 ; GPT-4o 75,32 / 75,43 / 73,46 ; SigLIP so400m 59,47 / 53,16 / 60,88 ; ResNet50 40,46.
  - Wikidata : SigLIP so400m 66,29 / 59,60 / 64,86 (meilleur) ; GPT-4o 45,75 / 45,31 / 45,31 ; Gemini 2.5 Pro 45,89 / 45,31 / 47,07 ; ResNet50 43,97. Chute attribuée par les auteurs à la taille d'image et à l'alignement des métadonnées.
  - Les descriptions Iconclass aident en général en zero-shot (surtout CLIP/SigLIP sur ArtDL) ; le few-shot est généralement inférieur au zero-shot ; piste : RAG. Mêmes sources.
- Cetinić, "Iconographic Image Captioning for Artworks" (arXiv 2102.03942, 07/02/2021, atelier FAPER/ICPR 2020) : annotations Iconclass converties en descriptions textuelles, modèle VL transformer fine-tuné ; légendes plus pertinentes pour l'histoire de l'art que des modèles entraînés sur images naturelles. Taille du jeu et métriques non lues sur la page abs (non vérifié) — [arXiv](https://arxiv.org/abs/2102.03942) ; version revue "Towards Generating and Evaluating Iconographic Image Captions of Artworks" (J. Imaging, 2021, avec métrique sans référence + évaluation qualitative) : [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8404909/)
- Springstein et al., "Visual Narratives: Large-scale Hierarchical Classification of Art-Historical Images", WACV 2024 : classification hiérarchique multi-label, jeu de 477 569 images, >20 000 concepts Iconclass ; les auteurs notent que les VLM n'avaient pas encore été appliqués à ce problème (PDF en 403, chiffres de résultats non vérifiés) — [CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Springstein_Visual_Narratives_Large-Scale_Hierarchical_Classification_of_Art-Historical_Images_WACV_2024_paper.pdf) (via résumé de recherche)
- ArtSeek (Fanelli, Vessio, Castellano, arXiv 2507.21917, juillet 2025) : ColQwen2 late-interaction retrieval + Qwen2.5-VL agentique, jeu WikiFragments ; revendique +8,4 % F1 en style vs GraphCLIP et +7,1 BLEU@1 en légendage sur ArtPedia (chiffres du résumé de recherche, non revérifiés sur l'abstract) ; pas d'évaluation iconographique spécifique — [arXiv](https://arxiv.org/pdf/2507.21917)
- A-MAR (Wang, ..., Worring, arXiv 2604.19689, 21/04/2026) : récupération multimodale pilotée par un plan de raisonnement ; benchmark diagnostique ArtCoT-QA ; évalué sur SemArt et Artpedia ; aucun chiffre sur la page abs — [arXiv](https://arxiv.org/abs/2604.19689)
- MMArt (Wang, ..., Worring, arXiv 2608.10706, v1 11/08/2026, v3 21/09/2026) : 74 234 peintures WikiArt, descriptions narrative/formelle/émotionnelle/historique ; retrieval R@1 = 44,0 % (narrative) vs 7,8 % (formelle) — [arXiv](https://arxiv.org/abs/2608.10706)
- GalleryGPT (ACM MM 2024, arXiv 2408.00491) : PaintingForm ~19k images / ~50k paragraphes d'analyse formelle générés par LLM sans voir l'image ; base ShareGPT4V-7B + LoRA ; les auteurs décrivent un "LLM-biased visual hallucination" qui nuit à la généralisation aux peintures sans nom (chiffres non vérifiés) — [arXiv](https://arxiv.org/pdf/2408.00491)
- ArtGPT-4 (arXiv 2305.07490, 2023) : MiniGPT-4 + adaptateurs ; résultats annoncés sur ArtEmis, à 0,15 point d'artistes sur une échelle de 6 (résumé de recherche, non revérifié) — [arXiv](https://arxiv.org/pdf/2305.07490)
- Strafforello et al., "Have Large VLMs Mastered Art History?" (arXiv 2409.03521, v1 05/09/2024, v2 10/09/2025) : CLIP, LLaVA, GPT-4o en zero-shot sur style/auteur/date ; chiffres non extraits (PDF illisible) — [arXiv](https://arxiv.org/abs/2409.03521)
- VULCA-Bench (arXiv 2601.07986, 12/01/2026) : 7 410 paires image-critique, 8 traditions, 5 niveaux ; niveaux L3-L5 (interprétation) plus difficiles que L1-L2 — [arXiv](https://arxiv.org/abs/2601.07986)
- Kutt & Baczyński, arXiv 2602.19698 (fév. 2026) : pipeline YOLOv8 + mapping Iconclass + règles, pas un benchmark VLM — [alphaxiv](https://www.alphaxiv.org/abs/2602.19698)

### Inferences
- Le bon score d'ArtDL (top-10 saints, attributs très codifiés) mesure surtout l'identification d'un personnage isolé ; il ne dit rien de la reconnaissance de relations ou de scènes.
- L'écart Wikidata vs ArtDL suggère une forte sensibilité au bruit/résolution et que SigLIP reste compétitif quand les MLLM n'ont rien de plus à "savoir".

### Gaps
- "RichArt" et "MARGE-GD" (ICMR 2026) : deux recherches n'ont rien retourné ; non vérifié, ne pas citer. À chercher dans le programme ICMR 2026 / ACM DL.
- "Saint George on a Bike" : seul un billet Europeana (projet de métadonnées par IA) est apparu ; aucune méthode ni chiffre iconographique vérifié.
- "ArtCap" : aucune source trouvée.
- Chiffres de Springstein WACV 2024, Strafforello, ArtGPT-4, GalleryGPT : non lus dans le texte intégral.
- Aucun benchmark 2024-2026 de VLM sur Iconclass pour scènes narratives trouvé.

## 2. Geste, pose et Pathosformel (panorama équilibré)

### Takeaway
La lignée pose-based va de Warburg/Impett (clusters de poses 2D, ambiguïté émotionnelle) à des jeux de données et estimateurs dédiés à l'art (PoPArt, Human-Art) ; le point commun est un domain gap marqué des estimateurs entraînés sur photos, et des représentations centrées sur des figures isolées plutôt que sur des relations.

### Cited Findings
- Impett & Süsstrunk, "Pose and Pathosformel in Aby Warburg's Bilderatlas", ECCV 2016 workshops : poses 2D annotées (crowdsourcing) sur ~un tiers des planches, angles relatifs des membres ; clusters correspondant à des Pathosformeln, mais des poses similaires peuvent exprimer des émotions très différentes (limite d'un encodage statique) — [mlanthology](https://mlanthology.org/eccv/2016/impett2016eccv-pose) ; résumé DH2017 : 1 665 figures en 16 clusters — [DH2017](https://dh2017.adho.org/abstracts/525/525.pdf) (via résumé de recherche)
- Jenicek & Chum, "Linking Art through Human Poses" (2019) : l'appariement explicite de poses surpasse la recherche par contenu standard sur un jeu annoté de transfert de composition (via résumé secondaire, non revérifié).
- Marsocci & Lastilla, POSE-ID-on (2021) : squelettes OpenPose, deux distances, K-Medians modifié — [lien](https://www.mdpi.com/2220-9964/10/4/257)
- Springstein, Schneider, Althaus, Ewerth, "Semi-supervised Human Pose Estimation in Art-historical Images", ACM MM 2022 (arXiv 2207.02976) : détection de personnes + keypoints semi-supervisés, jeu PoPArt ; "significativement meilleur" que modèles pré-entraînés/style transfer, sans chiffre sur la page abs — [arXiv](https://arxiv.org/abs/2207.02976)
- Schneider & Vollmer, "Poses of People in Art", ACM JOCCH 17(4), 2023 (jeu PoPArt) — référence citée par les résultats, page primaire non lue.
- Human-Art (Ju et al., CVPR 2023, arXiv 2303.02760) : 50k images, >123k personnes, 5 scénarios naturels + 15 artificiels ; les modèles existants échouent hors photos — [arXiv](https://arxiv.org/abs/2303.02760). ViTPose : 77,4 AP sur COCO vs 53,8 AP sur Human-Art (cité par LocLLM, source secondaire, non vérifié dans Human-Art) — [LocLLM](https://arxiv.org/pdf/2406.04659). VLPose : +2,26 AP sur l'état de l'art avec ViT-H, gains max ukiyo-e +8,2 — [arXiv](https://arxiv.org/pdf/2402.14456)
- Deliège & Dondero, "Similar Paintings Retrieval from Individual and Multiple Poses", ECCV workshops 2024 : extraction/normalisation de poses individuelles, comparaison de groupes de poses, corpus de peintures religieuses (donc prise en compte de plusieurs figures) — [mlanthology](https://mlanthology.org/eccvw/2024/deliege2024eccvw-similar)
- Impett & Süsstrunk, "Understanding Compositional Structures in Art Historical Images using Pose and Gaze Priors" (arXiv 2009.03807, 2020) — [arXiv](https://arxiv.org/pdf/2009.03807) (titre seulement, contenu non lu)
- Bernasconi, navigation par geste pour l'histoire de l'art (DH abstracts) — [lien](https://dh-abstracts.library.virginia.edu/works/11823) ; contenu non confirmé.
- MIRAGE (Chiu et al., arXiv 2604.23788, 26/04/2026) : représentation intermédiaire structurée (identités, indices de pose, hypothèses de regard) pour micro-interactions entre figures ; annonce moins d'hallucinations relationnelles vs VLM seuls, mais aucun chiffre sur la page abs — [arXiv](https://arxiv.org/abs/2604.23788)

### Inferences
- La plupart des travaux pose comparent des figures individuelles ; seuls Deliège & Dondero (poses multiples) et MIRAGE traitent explicitement l'interaction entre figures — la relation reste peu mesurée.

### Gaps
- Madhu et al., Zhao et al. : non retrouvés (recherches sans résultat) — non vérifié.
- Chiffres AP de PoPArt et ViTPose sur Human-Art dans le papier original non lus.

## 3. Descriptions de scène structurées (figures, rôles, relations) et hallucination sur images d'art

### Takeaway
Aucune évaluation rigoureuse et chiffrée des scene graphs d'œuvres par VLM n'a été trouvée ; les indices disponibles (MIRAGE, GalleryGPT, Fanelli 2026) pointent des hallucinations de relations/titres et un ancrage faible pour les termes abstraits.

### Cited Findings
- Fanelli et al., "Understanding How MLLMs Describe Artworks Using Token Activation Maps", arXiv 2606.27947, 26/06/2026 : cartes d'activation par token, 5 catégories (objets, style, métadonnées, iconographie, affect) ; tokens concrets (objets, iconographie) bien localisés, tokens abstraits diffus ; attribution d'artiste plus exacte que prédiction du titre, où les hallucinations sont plus fréquentes — [arXiv](https://arxiv.org/abs/2606.27947) (via résumé de recherche, chiffres non lus)
- MIRAGE (voir §2) : "relational hallucinations" réduites, qualitatif.
- Jiang & Chen (2025) : les MLLM hallucinent lors du raisonnement esthétique ; ArtCoT par preuves les réduit (référence secondaire, URL non obtenue — non vérifié).
- Recherche de scene graphs sur œuvres d'art : aucun résultat spécifique ; seulement des travaux sur images naturelles (ex. [arXiv 2509.01209](https://arxiv.org/pdf/2509.01209)).

### Inferences
- Les travaux art-spécifiques évaluent le texte (BERTScore, BLEU, R@1), pas des triplets sujet-relation-objet : une évaluation de graphes de scène d'œuvres reste à construire.

### Gaps
- Pas de benchmark d'hallucination d'objets/relations propre aux œuvres (POPE-art) trouvé. Absence de preuve, pas preuve d'absence.

## 4. Fuite de connaissance : reconnaître l'histoire connue plutôt que voir

### Takeaway
Indices convergents mais indirects que les VLM s'appuient sur le contexte linguistique/métadonnées plus que sur l'image ; aucune étude trouvée qui isole le "savoir de l'histoire" pour l'iconographie par contrefactuels.

### Cited Findings
- Kanazawa, Tanaka, Kera, Horiuchi, "Leveraging VLMs for Semantic Interpretation of Historical Paintings: A Case Study on Religiousness", Electronic Imaging 38(14), mars 2026 : peintures religieuses, BERTScore contre descriptions de musée, 3 conditions (image seule, image+métadonnées, métadonnées seules) ; "metadata improves scores, while visual input has limited impact" ; évaluation très dépendante des références — [IS&T](https://library.imaging.org/ei/articles/38/14)
- GalleryGPT teste sur peintures "sans nom" pour éviter le biais LLM (voir §1).
- Fanelli 2026 : titres plus hallucinés que les artistes (§3).
- Shahgir et al., "VLMs Need Words" (arXiv 2604.02486, COLM 2026) : les VLM comparent bien les entités nommables et produisent des descriptions fragiles/hallucinées pour les entités sans nom ; la page lue ne mentionne pas l'art — [arXiv](https://arxiv.org/abs/2604.02486). Un résumé de recherche lui attribuait un protocole "quel peintre ?" : non confirmé sur la page abs.
- Spinaci et al. : ajouter les descriptions Iconclass aide, le few-shot visuel nuit (§1) — compatible avec un poids important du texte/connaissance.

### Inferences
- Pour une étude propre : contrefactuels (œuvre célèbre vs copie/variante peu connue, masquage de titre/métadonnées, attributs modifiés) ; je n'ai pas trouvé d'étude le faisant en iconographie.

### Gaps
- Chiffres de Kanazawa et al. non lus. Pas d'étude "memorization/famous painting" iconographique primaire vérifiée.
