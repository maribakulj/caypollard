# Coût d'un ajustement de graphe de scène sur des œuvres (tâche 4)

Date : 17/10/2026. Règle : une URL par fait ; « non vérifié » quand je n'ai pas pu le lire dans le texte. Aucun calcul lancé. Textes extraits dans `src/couts/` (hors git).

## Réponse courte

- **Données** : rien de publié ne donne un nombre minimal d'images annotées en relations pour l'art. Le signal le plus solide vient de RelTR : il a été **ajusté sur VRD (4 000 images d'entraînement) à partir de poids Visual Genome, pendant 100 époques**. C'est le précédent le plus proche d'un petit jeu supervisé. https://arxiv.org/pdf/2201.11460 (texte extrait : `src/couts/reltr.txt`, section 4.3.1 / 4.2).
- **Calcul** : **aucun article ne rapporte un temps d'entraînement en heures**. RelTR rapporte seulement « 150 époques, 8 RTX 2080 Ti, batch 2 par GPU » pour VG et Open Images (même source, section 4.2). Le temps sur une T4 ou une P100 doit donc être **mesuré**, pas déduit des publications (voir c).
- **Annotation** : je n'ai **pas trouvé de coût publié par relation**. Le seul chiffre trouvé concerne les boîtes, pas les relations (voir d).

## a) Peu de données, légendes, transfert

| Article | Setting | Résultat | Données | Source |
|---|---|---|---|---|
| Zhong et al., ICCV 2021, *Learning to Generate Scene Graph from Natural Language Supervision* | entraînement sur paires image–légende (CC + COCO), sans graphe annoté | R@50 = **5,9** (R@100 = 7,0) sur VG ; **10,0** avec supervision faible sur VG ; **13,8** en supervision complète VG | CC + COCO ; VG pour les variantes faibles/complètes | https://arxiv.org/pdf/2109.02227 — chiffres lus dans un extrait de recherche, **à revérifier dans le PDF** (tableau 2 garbled) |
| Ye & Kovashka, CVPR 2021, *Linguistic Structures as Weak Supervision for Visual SGG* | supervision faible par structures linguistiques des légendes | non vérifié (chiffres non lus) | légendes | https://openaccess.thecvf.com/content/CVPR2021/html/Ye_Linguistic_Structures_As_Weak_Supervision_for_Visual_Scene_Graph_Generation_CVPR_2021_paper.html ; code https://github.com/yekeren/WSSGG |
| Kim et al., CVPR 2024, *LLM4SGG* | LLM pour extraire les triplets des légendes, puis entraînement faiblement supervisé | sur VG, VS³ : R@50 **14,54 → 18,40** (+3,86), mR@50 2,80 → 6,26 ; SGNLS : 3,80 → 5,09 ; Motifs entièrement supervisé (référence) : 31,89 | VG, GQA ; légendes COCO et CC | https://arxiv.org/pdf/2310.10404 — tableau 2 lu dans un extrait de recherche, **à revérifier** |
| Dornadula et al., *Learning Predicates as Functions to Enable Few-shot SGG* | few-shot (5 exemples par prédicat rare) | R@50 = **22,70** en 5-shot, +3,7 vs transfert classique | VG | https://arxiv.org/abs/1906.06798 — extrait lu, **non vérifié dans le PDF** |
| Chen et al., ICCV 2019, *Scene Graph Prediction with Limited Labels* | génération de données synthétiques | +8,27 points de R@100 vs transfert | VG | https://cs.stanford.edu/people/ranjaykrishna/limitedlabels/index.html — extrait lu, **non vérifié** |
| He et al., *Semantic Compositional Learning for Low-shot SGG* | prédicats rares/zéro-shot | chiffres R@50 **non vérifiés** | VG | https://arxiv.org/abs/2108.08600 |

Lecture : avec des légendes seules (sans boîtes), on passe d'environ 6 à 14 points de R@50 sur VG selon l'approche et la référence ; le plafond entièrement supervisé est autour de 31 à 32 (Motifs). **Aucun de ces articles ne traite de peintures ou d'estampes.** Pour une décision chiffrée sur l'art, il faut donc les tests sur nos propres images.

## b) Graphes appris à partir de légendes ou de descriptions

- Zhong et al. (ICCV 2021) : apprentissage à partir de paires image–description, sans graphe annoté : https://openaccess.thecvf.com/content/ICCV2021/html/Zhong_Learning_To_Generate_Scene_Graph_From_Natural_Language_Supervision_ICCV_2021_paper.html — résultats en (a).
- Ye & Kovashka (CVPR 2021) et Kim et al. (CVPR 2024) : voir (a).
- Pour RichArt et Iconclass (texte seul, pas de boîtes reliées) : la littérature trouvée est sur les légendes de photos. **Pas d'article trouvé sur des descriptions d'œuvres** (je n'ai pas cherché plus loin dans la limite de temps). Le jeu FRAME (texte d'art, relations typées, annotation manuelle) reste la référence textuelle la plus proche : https://arxiv.org/abs/2602.19133 (voir `verifications.md`).

## c) Ordre de grandeur du temps d'entraînement

Ce que disent les sources :
- **RelTR** : entraînement de bout en bout depuis zéro, **150 époques**, **8 RTX 2080 Ti**, batch 2 par GPU (donc batch effectif 16), AdamW, taux divisé par 10 après 100 époques ; **aucune durée totale en heures** (https://arxiv.org/pdf/2201.11460, section 4.2).
- **RelTR** : vitesse d'inférence **16,6 FPS** sur un RTX 2080 Ti, batch 1 (même article, section 4.3). Paramètres : **63,7 M** (section 4.1).
- **RelTR sur VRD** : fine-tuning à partir des poids VG, **100 époques**, avec les classifieurs de sujet, objet et prédicat réinitialisés (même article, section 4.2 ; VRD = 4 000 images d'entraînement).
- Pour le matériel : la carte **T4** (Kaggle) et la **P100** (Kaggle) n'ont pas été comparées à un 2080 Ti dans une source lue. **Comparaison de performance non vérifiée.**

Estimation (non sourcée, à mesurer) :
- Un fine-tuning sur **quelques milliers d'images** pendant **100 époques** représente au plus quelques centaines de milliers de passes d'image, contre environ **un ordre de grandeur de plus** pour un entraînement complet sur VG (taille de VG-150 : **non vérifié**, à lire dans le PDF).
- Méthode de mesure proposée : lancer une époque sur la T4 ou la P100 de Kaggle, mesurer le temps par image, puis multiplier par le nombre total de passes. Si une époque sur 3 000 images prend 5 min, 100 époques font environ 8 h, soit un peu plus d'une session Kaggle de 30 h/semaine. Ce chiffre est un exemple, pas une mesure.
- Sur Mac sans CUDA : RelTR nécessite la compilation de l'extension d'IoU dans `lib/fpn` (`make.sh`), donc **non lancable en local** (voir `modeles-sgg.md`) ; le calcul doit se faire sur Kaggle.

## d) Coût d'annotation rapporté

- **Visual Genome** (Krishna et al., IJCV 2017) : travail crowdsourcé sur Mechanical Turk ; les participants gagnaient **6 à 8 $ par heure** de travail continu ; 33 000 travailleurs uniques ; 50 descriptions de régions par image ; relations annotées par paires d'objets. **Pas de temps par relation ou par image rapporté** dans le texte lu. Sources : https://cs.stanford.edu/groups/vision/pdf/visualgenome.pdf ; texte extrait `src/couts/vg.txt`, section 4.1.
- **Temps de dessin d'une boîte** (non relation) : **42 s par boîte** (médiane), dont 25,5 s de dessin, 9 s de vérification de qualité et 7,8 s de vérification de couverture. Source citée : Su et al., 2012. Dans l'article de Pix2SG, cela donne **1 993 jours-personne** pour les 1,3 M boîtes de VG (environ 15 945 heures). https://arxiv.org/pdf/2303.10944 (table 7, section 4.5 ; texte `src/couts/lf.txt`). Ce chiffre ne concerne **que les boîtes**, il ne compte pas les relations ni les descriptions.
- **PSG** : **non trouvé** de coût publié (ni par image ni par triplet). Le site du jeu (psgdataset.org) n'a pas été ouvert.
- **SemArt-HOI** (mémoire Antoun, 2023) : annotation à la main par l'auteur, 73 683 boîtes et 40 101 étiquettes d'interaction, durée **non rapportée** dans le texte lu (voir `verifications.md`, source https://scholarworks.aub.edu.lb/bitstreams/11a7d9f5-0722-4a3d-8625-4f2b65a7c0b1/download).
- **Synthetic Visual Genome 2** (2026, vidéo) : annotation assistée par modèle, recrutement sur Prolific ; coût non trouvé. https://arxiv.org/pdf/2602.23543 (non ouvert).

## Constats principaux

1. **Données** : la littérature ne donne pas de seuil minimal publié pour des relations sur des œuvres. Le plus petit précédent supervisé est VRD (4 000 images) avec fine-tuning de 100 époques à partir de Visual Genome.
2. **Légendes** : sans boîtes, on passe d'environ 6 à 14 points de R@50 sur VG (LLM4SGG, Zhong et al.), contre environ 31 à 32 pour un modèle entièrement supervisé (Motifs). Aucun résultat sur les peintures.
3. **Calcul** : pas de durée publiée. Seule la mesure directe sur Kaggle (T4/P100) donnera un ordre de grandeur fiable.
4. **Annotation** : le coût publié est celui des boîtes (42 s par boîte, dont 25 s de dessin). Aucun coût par relation trouvé. Les salaires de crowdsourcing VG (6–8 $/h) donnent un ordre de grandeur indirect pour la main-d'œuvre.

## Sources

- RelTR : https://arxiv.org/pdf/2201.11460 ; dépôt https://github.com/yrcong/RelTR
- Zhong et al. 2021 : https://arxiv.org/pdf/2109.02227
- Ye & Kovashka 2021 : https://openaccess.thecvf.com/content/CVPR2021/html/Ye_Linguistic_Structures_As_Weak_Supervision_for_Visual_Scene_Graph_Generation_CVPR_2021_paper.html
- Kim et al. 2024 (LLM4SGG) : https://arxiv.org/pdf/2310.10404
- Dornadula et al. : https://arxiv.org/abs/1906.04876
- Chen et al. 2019 : https://cs.stanford.edu/people/ranjaykrishna/limitedlabels/index.html
- He et al. : https://arxiv.org/abs/2108.08600
- Visual Genome : https://cs.stanford.edu/groups/vision/pdf/visualgenome.pdf
- Pix2SG / coût des boîtes : https://arxiv.org/pdf/2303.10944
- SemArt-HOI : https://scholarworks.aub.edu.lb/bitstreams/11a7d9f5-0722-4a3d-8625-4f2b65a7c0b1/download
