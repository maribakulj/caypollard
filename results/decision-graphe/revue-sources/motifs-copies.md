# Revue : découverte de motifs et de copies dans les collections d'art (estampe -> majolique)

Date : 25/10/2026. Méthode : pages réellement ouvertes avec WebFetch (arXiv, ar5iv, GitHub, PMC, projets) ; les PDF illisibles ont été lus en local avec pdftotext quand ils avaient été téléchargés. Marques : [OUVERT] = page lue ; [RECHERCHE] = seulement un résumé de moteur de recherche, non vérifié ; [NON VÉRIFIÉ] = je n'ai pas pu confirmer. Aucun calcul lourd, aucun modèle téléchargé.

Attention : mes deux premières tentatives d'arXiv (IDs devinés) sont tombées sur des articles sans rapport ; tout ce qui suit utilise des IDs vérifiés par le contenu de la page.

## 0. Résumé court

- Aucune étude calculatoire sur Xanto ni sur les sources gravées de l'istoriato n'a été trouvée (recherches web, pas de résultat). Les notices de musées (Met, Fitzwilliam, Walters, Sotheby's, Pandolfini) citent Marcantonio/Raphaël comme sources, sans méthode automatique.
- Le plus proche en méthode : ArtMiner (Brueghel, détails copiés entre peintures/gravures), le moteur de recherche de détails de Ufer et al. (code sous licence GPL/MIT, état du dépôt à vérifier), SegSwap (poids publiés), et la recherche par poses de Jenicek & Chum (gère explicitement le miroir, mais code non annoncé).
- Aucun article trouvé qui évalue LoFTR, LightGlue, RoMa, DINOv2 dense, PE-Spatial ou C-RADIO sur des gravures/œuvres d'art. C'est un trou dans la littérature (au sens de ce que j'ai pu trouver) : il faudra le mesurer nous-mêmes.
- Le miroir est le point faible général : presque aucun article de l'art ne le traite, sauf Jenicek & Chum. Il faudra tester les deux orientations (on retourne la requête) pour tout système.

## 1. Découverte de motifs / copies dans les collections d'art

### 1.1 Shen, Efros, Aubry (ArtMiner), CVPR 2019
- URL ouvertes : https://arxiv.org/abs/1903.02678 et https://ar5iv.labs.arxiv.org/html/1903.02678 [OUVERT]
- Tâche : trouver des détails quasi dupliqués (copies avec imperfections, médias différents) dans une collection d'art.
- Données : Brueghel, 1 587 œuvres, 273 détails quasi dupliqués annotés (10 motifs) ; aussi Oxford5K (5 062 images), LTLL (225 historiques + 275 modernes), DocExplore (>1 500), WikiArt (Rubens 387, Rossetti 195, Canaletto 166).
- Scores (Table 1 de ar5iv, détection one-shot Brueghel, IoU>0,3, mAP) : ImageNet 58,0 / Context Prediction 58,8 / ArtMiner 75,3 (similarité cosinus) ; 76,4 avec le score de découverte. Video Google (SIFT) : 21,5 en retrait.
- Ce qui est apparié : région/détail (proposition de région, vérification Hough puis RANSAC affine) ; ResNet-18 conv4 affiné en auto-supervisé sur la collection (~10 h sur 1 GPU).
- Miroir : non mentionné (aucune augmentation ni appariement retourné). Échelle : 7 échelles sur 2 octaves.
- Passage à l'échelle : la découverte complète sur Brueghel a pris ~20 h sur 4 GPU ; les auteurs disent que cela limite l'usage à quelques milliers d'images.
- Code : l'article dit « Code and data is available at http://imagine.enpc.fr/~shenx/ArtMiner » (page ArtMiner du lab aussi listée par recherche : https://enherit.enpc.fr/artminer [RECHERCHE]). Je n'ai pas ouvert le dépôt ArtMiner lui-même [NON VÉRIFIÉ : poids].
- Note : la page arxiv.org/abs ne donnait pas de score ni de dépôt ; les chiffres viennent du texte ar5iv.

### 1.2 Shen, Efros, Joulin, Aubry (SegSwap), CVPR 2022 Workshops (IMW)
- URLs ouvertes : https://arxiv.org/abs/2110.15904 ; https://ar5iv.labs.arxiv.org/html/2110.15904 ; https://imagine.enpc.fr/~shenx/SegSwap/ ; https://github.com/XiSHEN0220/SegSwap [OUVERT]
- Tâche : co-segmentation (trouver les régions répétées entre deux images) pour la recherche de détails et la découverte, entraînée sur des paires synthétiques (segments COCO collés dans d'autres images, avec fusion de Poisson et transfert de style AdaIN entraîné sur Brueghel). Aucune annotation humaine.
- Données/scores : Brueghel, tableau 1 (ar5iv) : ArtMiner 75,5 (cosinus) ; MocoV2 79,0 ; SegSwap transformeur 81,8 (segments non supervisés) et 84,4 (segments COCO) ; Sparse-NCNet 82,8 et 83,3. Les en-têtes de colonnes du tableau ne se sont pas extraits proprement : je ne sais pas avec certitude quel nombre correspond à quelle métrique (recherche vs détection) [NON VÉRIFIÉ]. Aussi Tokyo247 et Pitts30K (reconnaissance de lieu, « compétitif »).
- Ce qui est apparié : régions entre deux images (masque + correspondances), paire par paire.
- Miroir : non mentionné dans l'article ni le dépôt. Échelle : l'article dit gérer de « grandes différences d'échelle » (qualitatif), augmentations rotation/translation/échelle/TPS à l'entraînement.
- Passage à l'échelle : mauvais. L'article reconnaît que la recherche score chaque paire d'images (limite) ; la découverte est quadratique (~10 h pour prédire toutes les paires de Brueghel, graphe de ~900 000 noeuds, 5 plus proches voisins par image), « pas passable à l'échelle ».
- Code/poids : code d'entraînement et d'évaluation (MIT), poids pré-entraînés MocoV2-ResNet50 + transformeur croisé (~300 Mo) via un script de téléchargement, jeu Brueghel dans le dépôt ; un second fichier (hard_mining_neg5.pth) sans source indiquée. Résolution en entrée non précisée ; Brueghel redimensionné à 640x640, patchs 320x320.
- Conséquence pour nous : utile comme revérificateur de paires (rerank d'un shortlist), pas comme moteur de recherche sur des milliers d'images.

### 1.3 Kaoua, Shen, Durr, Lazaris, Picard, Aubry, « Image Collation », ICDAR 2021
- URLs ouvertes : https://imagine.enpc.fr/~shenx/ImageCollation ; https://github.com/Rykoua/ImageCollation [OUVERT] ; arXiv 2108.08109 cité par la page projet et par la recherche (https://ar5iv.arxiv.org/html/2108.08109 listé, non ouvert) [RECHERCHE pour l'ID].
- Tâche : apparier les illustrations correspondantes entre manuscrits (alignement global d'illustrations, pas copie de figure isolée).
- Données : plus de 2 000 illustrations, ~1 200 correspondances annotées (6 manuscrits, 2 textes selon l'abstract ; la page du projet dit 3 textes et ~1 300 : incohérence signalée par la page).
- Scores : aucun chiffre sur la page projet [NON VÉRIFIÉ : tableaux de l'article non lus]. Idée : les mesures de similarité existantes marchent sur les cas simples et échouent sur les détails fins ; propagation par cycle-consistance donne des gains « significatifs ».
- Miroir : non mentionné. Code : dépôt public (PyTorch) ; poids et licence non indiqués.
- Utilité pour nous : faible directement (images entières par entières), idée de cycle-consistance utile pour fusionner des évidences.

### 1.4 Ufer, Simon, Lang, Ommer, PLOS ONE 2021 (version ECCV-VISART 2020 : Ufer, Lang, Ommer)
- URLs ouvertes : https://arxiv.org/pdf/2107.06935 (lu avec pdftotext) ; https://github.com/CompVis/visual-search ; (PMC derrière un captcha, non lu ; l'identité des auteurs/venue provient de la recherche : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8612525/ [RECHERCHE]).
- Tâche : recherche interactive d'objets/motifs (région requête) dans de grandes collections d'art, vote itératif sur des correspondances locales, fusion de caractéristiques multi-styles (transfert de style pour normaliser le domaine), index IVF+PQ pour la rapidité.
- Données/scores (Table 1 et 4 de l'arXiv, mAP, VGG16 dim 96) : Brueghel 85,7 (ArtMiner 79,9 ; ImageNet image entière 24,0) ; Brueghel5K 83,9 (ArtMiner 39,5) ; Brueghel101K 76,9 (ArtMiner 36,4) ; LTLL 90,9 ; Oxford5K 89,9. Brueghel5K et 101K sont des jeux ajoutés (WikiArt moins les Brueghel). Ablation : sans vote 72,3, avec un tour de vote 74,1, complet 85,7 (IoU 0,3). Aux IoU plus stricts : 63,3 (0,5) et 21,7 (0,7).
- Vitesse : temps de requête ~8,5-10,5 s selon les tailles testées (tableau avec 1,7 à 34,2 Go de mémoire, interprétation exacte de la taille d'index non lue) [partiellement vérifié].
- Ce qui est apparié : région requête (fenêtres glissantes multi-échelles) contre toute la collection ; vote de boîte, localisation.
- Miroir : non trouvé dans le texte (grep sur flip/mirror vide). Échelle : multi-échelle oui.
- Code : le dépôt GitHub CompVis/visual-search ne contient qu'un fichier LICENSE (MIT, 2020) et un commit : aucun code visible. L'article/recherche parle de code GPLv2 sur compvis.github.io/visual-search : incohérence entre les sources, je n'ai pas ouvert ce site [NON VÉRIFIÉ].

### 1.5 Seguin et al., Replica (EPFL DHLAB), 2016-2018
- Seguin, Striolo, di Lenardo, Kaplan, « Visual Link Retrieval in a Database of Paintings », ECCV-VISART 2016. Seulement la recherche : https://mlanthology.org/eccv/2016/seguin2016eccv-visual ; préprint listé : https://infoscience.epfl.ch/handle/20.500.14299/132112 (page vide à l'ouverture) [RECHERCHE]. Résultat annoncé : les CNN pré-entraînés surpassent les méthodes classiques, et un affinage améliore nettement.
- Seguin, di Lenardo, Kaplan, « Tracking transmission of details in paintings », DH2017 : https://dh2017.adho.org/abstracts/562/562.pdf [OUVERT, lu avec pdftotext]. Étend le moteur de recherche à l'appariement de détails (région requête, CNN affinés sur le jeu Replica) et dit explicitement qu'un caractère ou une tête a pu être copié d'un dessin, d'une gravure ou d'une tapisserie, donc le motif doit être détecté à travers médias et styles. Pas de chiffres ni de code dans ce texte ; pas de miroir.
- Ne pas confondre avec Seguin et al. 2020 « Visual link retrieval and knowledge discovery in painting datasets » (https://ar5iv.labs.arxiv.org/html/2003.08476 [OUVERT]) : 8 446 tableaux de 50 peintres (jeu Kaggle), VGG16, précision 72 % sur 100 liens jugés par 5 experts, pas de rappel, pas de code. Cet article porte sur les liens d'influence entre peintres, pas la copie de figures.

### 1.6 Jenicek & Chum, « Linking Art through Human Poses », ICDAR 2019
- URL ouverte : https://ar5iv.labs.arxiv.org/html/1907.03537 (arXiv 1907.03537) [OUVERT] ; PDF aussi lu en local pour le miroir.
- Tâche : lier des œuvres par la similarité des poses de figures (copie ou transfert de composition).
- Données : 635 images annotées (peintures, aussi dessins et gravures, surtout Renaissance et baroque) ; chaque paire marquée « copie » ou « transfert de composition » ; recherche qualitative sur ~37 000 images du Web Gallery of Art (sans score).
- Ce qui est apparié : poses OpenPose (BODY_25), normalisées sur le cou ; distance par cosinus ; deux distances d'image (meilleure paire de poses, ou appariement biparti approché), puis vérification géométrique RANSAC sur les 50 premiers (au moins 7 points clés alignés).
- Miroir : géré explicitement (distance minimale entre la pose et son miroir ; transformation estimée aussi sur la requête retournée). Le texte note qu'une estampe est inversée par rapport au tableau dont elle est tirée.
- Scores (mP@k, tous liens, Table I) : VGG GeM 56,8 / 51,0 / 41,7 / 34,6 / 26,9 (k=1,2,5,10,50) ; EdgeMAC 59,3 ... ; pose seule distmin 29,4 ; distt 50,1 ; avec vérification distmin 74,6, distt 75,3 (mP@1), mP@10 = 50,6, mP@50 = 40,9. Pour les « copies » seules : mP@1 87,7 contre 64,2 (GeM) et 71,0 (EdgeMAC). Pour le « transfert de composition » : 57,5 contre 41,0/40,4.
- Passage à l'échelle : appariement de poses linéaire, ~100 ms par requête sur 635 images ; vérification géométrique ~4,5 s par requête pour 50 candidats (Python non optimisé) ; pas de mesure sur 37 000 images.
- Code : pas annoncé dans l'article [NON VÉRIFIÉ ailleurs]. Limites : dépend de la détection des poses (OpenPose, entraîné sur photos) ; échoue pour figures partielles, drapées, très petites (déduction, non mesurée dans l'article).
- Pertinence : c'est le seul travail qui traite figure + miroir + gravure. Les scores sont mesurés sur des copies de peintures, pas des majoliques.

### 1.7 Impett & Süsstrunk (cité brièvement)
- « Pose and Pathosformel in Aby Warburg's Bilderatlas », ECCV 2016 Workshops : https://mlanthology.org/eccvw/2016/impett2016eccvw-pose [RECHERCHE]. Poses 2D annotées par crowdsourcing sur environ un tiers des planches, angles relatifs des membres, groupes de poses. Étude des Pathosformeln, pas un moteur de copie. Non ouvert plus loin.

### 1.8 Madhu et al., poses sur vases grecs
- Madhu, Villar-Corrales, Kosti, Bendschus, Reinhardt, Bell, Maier, Christlein, J. Computing and Cultural Heritage 2023 (venue/année par la recherche [RECHERCHE]) ; version arXiv ouverte : https://ar5iv.labs.arxiv.org/html/2012.05616 [OUVERT].
- Données : ClassArch, 1 513 images (2 629 boîtes de personnes, 1 728 poses) ; l'article incohérent avec lui-même (1 783 images dans le texte) ; COCO-Persons stylisé (66 808 images).
- Scores : estimation de pose sur ClassArch mAP 24,7 (base) vers 32,3 (transfert de style), mAR 30,9 vers 39,0 ; détection de personnes 10,4 vers 49,4 (modèles affinés). Recherche de poses sur 303 images de validation : P@1 caractères 31,7 vers 47,7 ; scènes 43,6 vers 58,8.
- Miroir non traité. Code : https://github.com/angelvillar96/STLPose (cité par la page ; contenu du dépôt non ouvert). Jeu de données stylisé « sera publié avec le code ».
- Limite : petite taille d'évaluation, vases (silhouettes) loin de la majolique polychrome.

### 1.9 Springstein, Schneider, Althaus, Ewerth, ACM MM 2022
- URLs ouvertes : https://ar5iv.labs.arxiv.org/html/2207.02976 ; https://github.com/tibhannover/iart-semi-pose [OUVERT].
- Tâche : estimation de pose humaine semi-supervisée sur images d'art (détection de personnes puis points clés, enseignant-élève).
- Données : PoPArt, 2 859 images, 3 514 personnes, 51 645 points clés (train 1 553 / val 643 / test 663 images), 22 styles selon la recherche [RECHERCHE pour « 22 styles »].
- Scores : AP points clés sur test 0,5258 (méthode finale avec boîtes prédites) contre 0,2525 (meilleur COCO avec transfert de style) et 0,1388 (OpenPose) ; avec boîtes de référence 0,7592 ; détection de personnes AP 0,5073 contre 0,3846. Recherche de pose : étude utilisateur sur ART500k, NDCG, marges plus faibles.
- Code et poids : oui (poids sur Hugging Face springsteinm/iart-semi-pose) ; pas de licence indiquée.
- Pertinence : le meilleur détecteur de poses d'art disponible, bon candidat si on veut une comparaison de poses à la Jenicek & Chum.

### 1.10 Autres pistes trouvées
- Zhou & Kong, « Dare to Plagiarize? Plagiarized Painting Recognition and Retrieval » (https://arxiv.org/abs/2506.23132 [OUVERT] ; 2025) : DINOv2 brut 97,2 % d'exactitude de reconnaissance mais seulement 29,0 % d'AP en recherche ; affinage métrique +12 points d'AP (41,2 %) mais exactitude 92,7 %. Les « plagiats » sont générés par IA, pas des copies historiques ; code non indiqué. Seule valeur : montre que les descripteurs globaux DINOv2 sont mauvais pour classer les copies.
- PrintArt (Lisbonne : IST, Faculdade de Letras, Museu Nacional do Azulejo) : https://sipg.isr.tecnico.ulisboa.pt/printart/ [OUVERT]. Projet de logiciel pour apparier des panneaux d'azulejos aux estampes sources ; la page ne donne ni méthode, ni résultat, ni date ; elle mentionne que les artisans inversaient les figures. Analogie directe avec notre problème, mais pas de sortie évaluable.
- Imagematching / VISE (Oxford VGG) pour les illustrations de livres imprimés renaissants, d'après des résumés de recherche (talk Tramelli, déc. 2025 ; Biblissima) : [RECHERCHE], pages non ouvertes. À vérifier : https://www.uibk.ac.at/media/filer_public/d6/f4/d6f471d0-b2ee-45ef-8b64-f25703e36f95/tramelli_dez_2025.pdf.
- Projets ornements / bois gravés réutilisés (Wilkinson et al., DHQ 2021 ; IPOP Utrecht) : résumés de recherche seulement [RECHERCHE].
- Non trouvé : « Fragmentation », « Les Dossiers », un travail 2024-2026 de l'équipe d'Aubry sur la copie estampe-peinture, tout article sur Xanto ou la majolique avec vision par ordinateur. Absence de résultat de recherche, pas preuve d'absence. [NON VÉRIFIÉ]
- Sources documentaires utiles pour la vérité terrain (pas du calcul) : notices Fitzwilliam, Met, Walters, Sotheby's, Pandolfini retrouvées par la recherche (par ex. https://www.metmuseum.org/art/collection/search/198702 ; https://art.thewalters.org/detail/37839/ ; Pandolfini donne La Fortezza, Bartsch XV n.14) [RECHERCHE, non ouvertes].

## 2. Appariement partiel région-vers-image robuste au changement de style

### 2.1 Correspondances locales
- RoMa (Edstedt et al., CVPR 2024) : https://github.com/Parskatt/RoMa [OUVERT]. Code `romatch` MIT, repose sur DINOv2 (Apache 2.0) pour le grossier ; résolution par défaut 560x560 montée à 864x864 ; pas de chiffres dans le README pour RoMa lui-même (seulement Tiny RoMa : AUC@5 56,4 sur Mega1500 contre 46,4 XFeat). Je n'ai pas ouvert l'article (ID 2305.15404 donné par le README).
- LightGlue (Lindenberger et al., ICCV 2023) : https://github.com/cvg/LightGlue [OUVERT]. Poids pour SuperPoint, DISK, ALIKED, SIFT ; Apache-2.0 sauf SuperPoint (licence restrictive, y compris poids) ; 150 FPS à 1024 points clés sur RTX 3080, ~20 FPS à 512 points sur CPU i7. C'est un matcheur de paires, pas un moteur de recherche.
- LoFTR : non ouvert [NON VÉRIFIÉ].
- Aucune évaluation sur des œuvres d'art : recherche ciblée sans résultat. Les repères indirects (CM-Bench visible/infrarouge, tableau « Zeb » non identifié, AUC@5 : 53,3 GIM-RoMa, 33,1 LoFTR, 31,7 LightGlue) viennent de résumés de recherche [RECHERCHE] et ne se transfèrent pas à l'art. Hypothèse (non testée) : les matcheurs denses à base de DINOv2 résistent mieux au changement d'apparence que SuperPoint, mais la gravure en traits (hachures) contre une peinture polychrome est un cas atypique.
- Le modèle SIFT + Bag-of-Words reste cité par Seguin et al. comme état de l'art pour photos, mais s'effondre sur les changements de médium (ArtMiner : SIFT 21,5 mAP).

### 2.2 Descripteurs denses (DINOv2, PE-Spatial, C-RADIO)
- DINOv2 : https://arxiv.org/pdf/2304.07193 (résumé de recherche seulement : transfert qualitatif en profondeur/segmentation sur peintures, pas d'évaluation d'appariement) [RECHERCHE].
- PE-Spatial (Perception Encoder, Meta) : https://arxiv.org/pdf/2504.13181 ; Table 13 : prédiction dense gelée (suivi, segmentation, profondeur) [RECHERCHE], aucune évaluation sur l'art trouvée. Le résumé tiers « surpasse DINOv2 » vient d'un README, non confirmé.
- C-RADIO / RADIOv2.5 : supplément CVPR 2025 https://openaccess.thecvf.com/content/CVPR2025/supplemental/Heinrich_RADIOv2.5_Improved_Baselines_CVPR_2025_supplemental.pdf [RECHERCHE] ; aucune évaluation sur l'art trouvée.
- Aucune comparaison directe PE-Spatial / RADIO / DINOv2 sur gravure vers peinture n'existe à ma connaissance après ces recherches [NON VÉRIFIÉ au sens strict].
- Un travail récent d'enquête (Du et al., « Cross-View Feature Matching: Survey, Benchmarking... », arXiv 2608.11093, août 2026) [OUVERT, abstract seulement] cite la généralisation inter-domaines comme défi ouvert ; ne dit pas qu'il évalue des œuvres d'art.

## 3. Tableau de synthèse

| Travail | Unité appariée | Miroir | Mille+ images | Code / poids vérifiés |
|---|---|---|---|---|
| ArtMiner 2019 | détail (région) | non mentionné | découverte limitée à quelques milliers | code annoncé par l'article, dépôt non ouvert |
| Ufer 2021 | région requête | non trouvé | oui (101K, ~9 s/requête) | dépôt vide (LICENSE seul) |
| SegSwap 2022 | paire d'images, régions | non mentionné | non (quadratique) | code MIT + poids oui |
| Image Collation 2021 | illustration entière | non mentionné | non évalué | code oui, poids/licence non |
| Seguin 2017 (Replica) | détail | non | oui (moteur) | non indiqué |
| Jenicek & Chum 2019 | poses de figures | oui, explicite | pas mesuré (100 ms/requête sur 635) | non annoncé |
| Madhu 2023 | pose (vases) | non | non (303 images) | GitHub cité, non ouvert |
| Springstein 2022 | pose (art) | non | n/a | code + poids oui |
| RoMa / LightGlue | paire d'images | à gérer soi-même | paire par paire | oui |

## 4. Ce que cela implique (jugement, pas fait vérifié)

1. Pas d'état de l'art prêt à l'emploi pour « figure d'une gravure retrouvée sur une majolique ». Les travaux les plus proches ont été évalués sur des peintures de la famille Brueghel, des manuscrits, et des collections WikiArt, pas des céramiques.
2. Le miroir devra être traité à la main (requête et son retournement) ; seul Jenicek & Chum l'ont fait dans le papier.
3. Les nombreuses copies quasi identiques d'une même estampe par des graveurs différents ne sont traitées dans aucun papier trouvé ; ce sera à notre graphe de regrouper les variantes avant de classer.
4. Un pipeline plausible : recherche de candidats par caractéristiques denses sur des vignettes de figure, puis re-vérification géométrique (RoMa ou SegSwap) sur un petit nombre de candidats. À valider par une mesure sur nos propres paires, car aucune des briques n'a de chiffre sur ce domaine.

## 5. Non vérifié / à rouvrir
- PMC de Ufer (captcha), dépôt compvis.github.io/visual-search, site/dépôt ArtMiner, dépôt STLPose.
- Tableaux complets de Image Collation et SegSwap (colonnes ambiguës).
- Article de RoMa et LoFTR, DINOv2, PE, RADIO : seulement README ou résumé de recherche.
- Toute publication 2024-2026 sur la copie estampe vers peinture/céramique : non trouvée, ce n'est pas la preuve qu'elle n'existe pas.
