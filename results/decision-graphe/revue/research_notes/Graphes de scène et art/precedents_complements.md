# Précédents directs : compléments (état au 17/10/2026, EN COURS)

Suite de precedents_directs.md. Les items sont ajoutés au fur et à mesure.

## 1. DEArt, extension 2025 (Journal of Cultural Heritage) : relations ?

### Takeaway
L'extension existe et annonce explicitement « un approche pour générer des labels d'objets et de RELATIONS raffinés sans annotation manuelle » ; ce sont donc des relations générées automatiquement (pas annotées à la main). Je n'ai PAS pu lire le texte intégral (ScienceDirect 403) : vocabulaire de relations, taille et disponibilité des relations restent non vérifiés.

### Cited Findings
- Référence : Reshetnikov, Marinescu, More Lopez, Mendoza, Freire, Marrero, Tsoupra, Isaac, « DEArt: Building and evaluating a dataset for object detection and pose classification for European art », Journal of Cultural Heritage, vol. 75, pp. 258-266, 2025 (vor en ligne 06/08/2025 d'après Crossref ; daté oct. 2025 par le dépôt URL). DOI 10.1016/j.culher.2025.07.022. — [Crossref](https://api.crossref.org/works/10.1016/j.culher.2025.07.022) ; [dépôt Univ. Ramon Llull](https://merit.url.edu/ca/publications/deart-building-and-evaluating-a-dataset-for-object-detection-and-/)
- Résumé (texte exact du dépôt) : l'article étend le travail de 4 façons : (1) approche pour générer des labels d'objets et de relations raffinés sans annotation manuelle ; (2) comparaison avec l'état de l'art en vision et patrimoine ; (3) évaluation par des utilisateurs des annotations générées par le modèle de détection (correction et pertinence) ; (4) brève discussion d'équité (fairness) du jeu. Mots-clés : dataset/model fairness, object detection, pose classification, iconographic art. — [dépôt URL](https://merit.url.edu/ca/publications/deart-building-and-evaluating-a-dataset-for-object-detection-and-/)
- Licence de l'article : CC BY-NC-ND 4.0 (version of record, Crossref) donc accès libre ; le texte intégral reste à lire pour la section relations. — [Crossref](https://api.crossref.org/works/10.1016/j.culher.2025.07.022)
- Jeu de données d'origine (v1, 21/10/2022) sur B2SHARE/EUDAT : « plus de 15 000 images, 69 classes, 12 poses » ; licence CC BY-NC-SA 4.0 ; 17 fichiers, 18,16 Go au total (annots_pub.zip = 10 Mo d'annotations + images_pub_01..16.zip). Aucune mention de relations dans la description de cette notice. — [B2SHARE k23em-7tn08](https://b2share.eudat.eu/records/k23em-7tn08) (métadonnées lues via l'API https://b2share.eudat.eu/api/records/k23em-7tn08)
- arXiv 2211.01226 : v1 02/11/2022, v2 03/11/2022 ; pas de lien jeu/code ; CC BY 4.0 pour l'article ; pas de DOI de la version journal affiché. — [arXiv](https://arxiv.org/abs/2211.01226)
- Autre extension, distincte (ECCV AI4DH workshop 2024) : Yemelianenko et al., « An Approach for Dataset Extension for Object Detection in Artworks Using Open-Vocabulary Models » : étend DEArt de 97,2 % et +12,2 % de mAP0.5 en moyenne ; ajoute WikiArt/Ukiyo-e ; objets seulement (résumé de moteur de recherche, page non ouverte). — [MLAnthology](https://mlanthology.org/eccvw/2024/yemelianenko2024eccvw-approach)

### Inferences
- Les relations de l'extension 2025 sont a priori générées automatiquement (et évaluées par utilisateurs), donc plus proches d'un pseudo-label que d'une vérité terrain ; la version publiée sur B2SHARE (2022) n'en contient pas d'après sa description.
- Il est possible que les relations soient un type de relation « avec ontologie/graphe de connaissances » (équipe Europeana/BSC) ; non vérifié.

### Gaps
- Texte intégral (vocabulaire de relations, taille, données de disponibilité, éventuel nouveau dépôt) : non lu (ScienceDirect 403, hdl URL.edu « Access Denied »). À ouvrir dans un navigateur ou par courriel aux auteurs.
- Aucun Zenodo/GitHub trouvé pour les relations.

## 2. Antoun, version article de SemArt-HOI

### Takeaway
La version article est paru dans Digital Applications in Archaeology and Cultural Heritage (Elsevier), vol. 34, article e00364, sept. 2024 (DOI 10.1016/j.daach.2024.e00364), auteurs Antoun et Asmar. Publication des données : non vérifiée (texte intégral non lu), et la thèse ne donnait ni URL ni licence.

### Cited Findings
- Titre « Human object interaction detection in paintings using multi-task learning », DAACH vol. 34, e00364, 09/2024, auteurs Antoun, Asmar. — [Crossref](https://api.crossref.org/works/10.1016/j.daach.2024.e00364) ; [lien DOI](https://doi.org/10.1016/j.daach.2024.e00364)
- Le PII ScienceDirect S2212054824000493 de la note précédente correspond a priori à cet article (même titre) ; non confirmé page par page (403). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2212054824000493)

### Gaps
- Données publiées ? Non établi (aucun Zenodo/GitHub trouvé dans les recherches).

## 3. Travaux récents sur les relations entre figures (2026)

### Takeaway
Deux préprints 2026 sont des précédents directs récents, mais aucun ne publie (à ma connaissance) de jeu de données : CompoVista/VisTCP (peinture chinoise, 6 180 relations annotées, 37 types) et MIRAGE (interactions entre figures dans 15 images, évaluation par VLM).

### Cited Findings
- CompoVista (arXiv 2607.07105 ; v1 08/07/2026) : 960 peintures chinoises traditionnelles (Tang à moderne, rouleaux verticaux/albums/éventails, source CAOD), 7 587 entités, 6 180 relations, 589 « vides » ; 37 types de relations en deux groupes (localisation : above, behind, inside ; événement : holding, sitting on, watching) ; 3 annotateurs (2 experts + 1 co-auteur), accord vérifié sur 120 peintures : Cohen kappa 0,86 (labels), IoU moyen 0,78 (régions) ; disponibilité des données/code non indiquée dans la partie lue (60 % du texte non lu). — [arXiv HTML](https://arxiv.org/html/2607.07105)
- CONTRADICTION : l'agrégateur Pith donne « 1 197 relations (≈1,25 par peinture) » ; le texte de l'article donne 6 180. L'article prime. — [Pith](https://pith.science/paper/2607.07105)
- VisTCP (arXiv 2607.05841, v1 07/07/2026, cs.HC/cs.AI, CC BY-NC-ND 4.0) : taxonomie sémantique établie avec 3 experts, modèle entraîné sur annotations d'experts pour extraire objets et relations, vue d'embedding commune annotations/prédictions, boucle experte ; aucun nom/lien de jeu ni de code. — [arXiv](https://arxiv.org/abs/2607.05841)
- MIRAGE (arXiv 2604.23788 ; v1 26/04/2026 ; Chiu, Wang, Luo, Wang, Yang, Khanal, Chen) : cadre « evidence-centric » pour micro-interactions (regard, geste, arrangement spatial) dans des œuvres multi-figures ; évaluation sur 15 images seulement (5 tableaux canoniques, 5 moins connus, 5 scènes générées ; 3-5 personnages, 5-7 interactions annotées par image) ; annotation : existence (oui/non/ambigu), type (support, guidance, attention partagée), direction ; désaccords réglés par consensus ; baseline GPT-5.4 image seule vs MIRAGE (image + image annotée + document de grounding) : identité 0,72 vs 0,92 ; interaction 0,81 vs 0,94 ; direction 0,83 vs 0,92 ; grounding 0,71 vs 0,88 ; pas de variance ni de test ; le nombre d'items est un placeholder « NN » dans le texte ; pas d'ontologie formelle ni de graphe de scène ; pas de publication de code/données indiquée. — [arXiv abs](https://arxiv.org/abs/2604.23788) ; [arXiv HTML](https://arxiv.org/html/2604.23788)

### Inferences
- Les 15 images de MIRAGE sont un test qualitatif, pas un benchmark. Les relations de CompoVista sont les plus structurées trouvées (vocabulaire à 37 types, kappa 0,86) mais hors art européen et sans publication établie.

### Gaps
- Disponibilité des données CompoVista : fin du texte non lue.

## 4. VISART / CHR / DH / JOCCH : travaux sur relations, gestes, narration

### Takeaway
Aucun article VISART 2026 listé n'a un titre sur les relations/scene graphs entre figures ; seul « TinTin DCU Benchmark: Page-level Comic Narrative Understanding » touche la narration (bande dessinée). Recherche CHR 2022-2025, DH, DSH, JOCCH non exhaustive : aucun article pertinent n'a été trouvé, ce qui n'est PAS une preuve d'absence.

### Cited Findings
- Programme VISART 2026 (8e édition) : 4 exposés + posters ; titres pertinents au domaine : « TinTin DCU Benchmark: Page-level Comic Narrative Understanding », « MMArt: A Multi-Perspective Multimodal Dataset for Visual Art Understanding », « Learning visual representations for compositional analysis of artworks and photographs », « Benchmarking Vision-Language and Multimodal Language Models For Cultural Heritage: An Analysis of Christian Iconography Classification » ; aucune page de programme 2024 ou 2025 lue (le site ne les liste pas). Seuls les titres ont été lus. — [visarts.eu](https://visarts.eu/)
- MMArt (arXiv 2608.10706, 2026) : 74 234 peintures, une légende unique combinant 4 perspectives (narrative, formelle, émotionnelle, historique) ; d'après un extrait de recherche (section travaux connexes seule) ; annotation, licence, présence de relations explicites NON VÉRIFIÉS. — [arXiv](https://arxiv.org/pdf/2608.10706)
- SensoryArt / SniffyArt (Zinnen, Hussian, Maier, Christlein), « Recognizing sensory gestures in historical artworks », Multimedia Tools and Applications vol. 84(32), 2024, DOI 10.1007/s11042-024-20502-6 : boîtes de personnes + 17 keypoints + label de geste (gestes sensoriels) ; SniffyArt : 1 941 personnes dans 441 œuvres. Geste par individu, pas de relation entre figures ; taille de SensoryArt non vue ; notice FAU sous licence Creative Commons (version non précisée). — [FAU](https://open.fau.de/handle/openfau/37753) ; [SniffyArt](https://cris.fau.de/publications/313446862)
- Bernasconi (UZH) : travail sur la reconnaissance automatique des gestes de mains en peinture et sa relation à l'iconographie (court article DH et article MDPI J. Imaging 9(6):120) ; contenus non ouverts. — [DH abstracts](https://dh-abstracts.library.virginia.edu/works/11823) ; [MDPI](https://www.mdpi.com/2313-433X/9/6/120)
- Bande dessinée / comics (domaine voisin) : CoMix (NeurIPS 2024, arXiv 2407.03550) : détection, locuteur, ordre de lecture, noms de personnages ; ComicScene154 ; MangaVQA ; arXiv 2603.01950 (2026) critique la similarité sémantique comme mesure de compréhension de pages de BD. Contenus non ouverts. — [CoMix](https://arxiv.org/pdf/2407.03550) ; [2603.01950](https://arxiv.org/pdf/2603.01950)
- Appel à contributions (arthist.net, daté 09.05.2026) pour un numéro thématique de Computational Humanities Research (Cambridge) sur art et culture visuelle : indique que la revue accueille ce type de travaux ; sans liste d'articles. — [arthist.net](https://arthist.net/archive/52416)
- Recherches sur SGG/VRD « paintings/artworks » 2025 : uniquement SGG sur images naturelles. — requêtes du 17/10/2026, aucun lien d'article art.

### Inferences
- Les relations entre figures en art sont traitées par proxys (gestes individuels, poses, regards, graphes de composition chinois, évaluation par VLM) plutôt que par un benchmark de triplets sujet-prédicat-objet.

### Gaps
- Actes VISART 2022-2025, CHR 2022-2025 (CEUR-WS), DH, DSH, JOCCH, IJDAR : non parcourus un par un.
- TinTin DCU : aucune source trouvée (existence non vérifiée en dehors du programme VISART 2026).

## 5. Jeux de données non encore examinés : relations, licence, disponibilité

### Takeaway
Aucun de ces jeux ne contient de relations entre figures ; tous sont objets/pose/classes. Les plus utilisables (licence ouverte vérifiée) : PoPArt (CC BY 4.0), IconArt (CC BY 4.0), Iconclass AI Test Set (CC0), ArtDL (Zenodo).

### Cited Findings
- PoPArt (Schneider, Vollmer ; arXiv 2301.05124, 12/01/2023) : 17 keypoints + boîte par figure, 22 styles, max 4 figures annotées par image, splits 1 553/643/663 images ; Zenodo 7516230 v1.1 (01/01/2023), CC BY 4.0, PoPArt.zip ≈ 973 Mo. Pas de relations. (Le DOI JOCCH 10.1145/3696455 de la note précédente reste non vérifié.) — [Zenodo](https://zenodo.org/records/7516230) ; [arXiv](https://arxiv.org/pdf/2301.05124)
- IconArt (Gonthier et al., ECCV 2018 VISART) : 5 955 images WikiCommons, 1 480 des 2 977 images de test annotées en boîtes pour 7 classes (angel, Child_Jesus, crucifixion_of_Jesus, Mary...) ; Zenodo 4737435 (05/10/2018), CC BY 4.0 ; pas de relations. — [Zenodo](https://zenodo.org/records/4737435)
- ArtDL (Milani, Fraternali) : 42 479 peintures, 19 classes Iconclass (saints, personnages chrétiens), Zenodo 6473001 v2.0 (16/07/2021), métadonnée CC BY 4.0 (résumé de recherche dit « recherche académique seulement » : à reconfirmer) ; classification, pas de relations ; attribution aux auteurs non confirmée. — [Zenodo](https://zenodo.org/records/6473001)
- Iconclass AI Test Set (Posthumus, 21/02/2020) : 87 749 images (≤ 500 px), notations Iconclass par image (JSON), CC0, zip 3,1 Go ; images de la base Arkyves ou partenaires sous licence ouverte ; notations seules. Les notations Iconclass peuvent coder des scènes/actions (ex. codes de scènes bibliques) mais ne donnent pas de triplets. — [iconclass.org/testset](https://iconclass.org/testset/)
- PeopleArt (Westlake, Cai, Hall ; arXiv 1610.08871, 2016) : détection de personnes dans l'art, boîtes ; environ 4 821 images, 43 styles (chiffres de résumés de recherche, NON VÉRIFIÉS) ; pas de relations. — [arXiv](https://arxiv.org/pdf/1610.08871)
- HORAE (Boillet et al., arXiv 2012.00351) : annotations de mise en page et de texte pour des livres d'heures (500 manuscrits numérisés, BVMM/Gallica) ; aucune annotation de figures ni de relations d'après les résultats vus. — [arXiv](https://arxiv.org/pdf/2012.00351)
- Emblematica Online : métadonnées liées (≈ 12 300 emblèmes dans 220 livres exposés en LOD ; la collection compte >1 400 livres et >25 000 emblèmes selon un résumé DH), ontologie RDF CC0 ; aucun jeu de détection ou de relations visuelles trouvé. — [DH abstracts 1989](https://dh-abstracts.library.virginia.edu/works/1989) ; [ontologie HAB](https://git.hab.de/dmj/uri.hab.de/-/raw/ece5db30843d372bd97f5a1f6b0261828165eaba/public/ontology/emblem.rdf)
- VQArt-Bench (arXiv 2510.12750, v1 14/10/2025) : VQA culturel généré par pipeline multi-agents, 14 MLLM évalués, « relations visuelles complexes » citées dans l'abstract ; nombre de questions, catégories, licence, lien de données non indiqués sur la page arXiv (le chiffre 14 463 de la note précédente reste non vérifié). — [arXiv](https://arxiv.org/abs/2510.12750)
- Iconographic Image Captioning for Artworks (arXiv 2102.03942) : légendes à partir d'Iconclass AI Test Set (86 530 images valides utilisées) ; contenu relationnel des légendes non vérifié. — [arXiv](https://arxiv.org/pdf/2102.03942)

### Inferences
- Pour un SGG art-historique, les jeux sous licence ouverte fournissent des figures/poses/classes (PoPArt, IconArt, DEArt NC) que l'on peut combiner, mais jamais la vérité terrain de relations.

### Gaps
- MET Open Access avec descriptions : non recherché (aucun appel) ; ArtPedia, Brueghel, Mandragore toujours non traités.
- PeopleArt : chiffres non vérifiés sur page primaire.
- Licence réelle d'ArtDL (CC BY vs usage académique) à clarifier.
