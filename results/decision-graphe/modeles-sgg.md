# Modèles de génération de graphe de scène avec poids publics (tâche 2)

Date des recherches : 17/10/2026. Les champs marqués **non vérifié** n'ont pas pu être lus dans les pages consultées.

Contrainte Mac (sans CUDA) : la plupart des dépôts anciens demandent une compilation CUDA ou des versions PyTorch/CUDA précises. Ce point est indiqué pour chaque modèle.

## Synthèse

| Modèle | Poids publics | Licence | Paramètres | Taille poids | Données | Relations | Dernière activité | Mac sans CUDA |
|---|---|---|---|---|---|---|---|---|
| RelTR (ECCV 2022, TPAMI 2023) | oui, Google Drive (liens dans le README) | **non vérifié** (pas de fichier LICENSE vu) | **63,7 M** | non vérifié | Visual Genome, Open Images V6 | **fermé** : 50 prédicats (VG) | non vérifié (44 commits, pas de date) | **probablement non** : PyTorch 1.6 / CUDA 10.1 + `make.sh` (IoU compilé) |
| OpenPSG (ECCV 2024, open-set) | **non trouvé** (aucun checkpoint cité) | non vérifié | non vérifié | — | PSG (`psg_train.json`, `psg_val.json`) | ouvert (LMM) — **non vérifié** | non vérifié (7 commits) | non, mmdet/openseed + entraînement 8 GPU |
| OvSGTR (mai 2025) | oui, Google Drive + Hugging Face JosephZ/OvSGTR | **Apache-2.0** | non vérifié | non vérifié (pas de taille affichée) | VG (pré-entraînement), COCO, Action Genome (fine-tuning) | **ouvert** (vocabulaire ouvert, catégories non vérifiées) | non vérifié (51 commits) | **difficile** : PyTorch 1.9.1 + CUDA 11.1, compilation GroundingDINO |
| OvDSGG (repo HF jhelsby, lié à OvSGTR) | oui, poids sous `exps/*/checkpoint.pth` (pas de lien direct vu) | **Apache-2.0** | non vérifié | non vérifié | Action Genome (vidéo, graphes dynamiques) | ouvert (25 objets / 18 prédicats d'entraînement, 36/26 au total) | non vérifié | **difficile** : Python 3.9 + install.sh |
| Modèles Video-LLaVA SGG (Aditya02, HF) | oui | non vérifié | non vérifié | non vérifié | Vidor (une variante) | non vérifié | 13-15/02/2025 | non vérifié (vidéo, probablement GPU) |

## Détail par modèle

**1. RelTR — Relation Transformer for Scene Graph Generation**
- Poids : README du dépôt officiel https://github.com/yrcong/RelTR (vu via le site : `ckpt/checkpoint0149.pth` pour VG, `ckpt/checkpoint0149_oi.pth` pour Open Images V6, liens Google Drive).
- Paramètres : **63,7 M** (article, section IV-C1 : « which has only 63.7M parameters » — https://arxiv.org/html/2201.11460). Taille du fichier : non vérifié.
- Vocabulaire : VG split : 150 catégories d'entités et **50 prédicats** (article, section IV-A1).
- Vitesse : 16,6 FPS sur un RTX 2080 Ti (article, IV-C1) ; le tableau I donne 16,1 FPS : chiffres légèrement incohérents dans l'article.
- Licence : **non vérifié**. Aucun fichier LICENSE vu sur le dépôt ; à confirmer avant usage.
- Environnement : Python 3.6, PyTorch 1.6.0, torchvision 0.7.0, cudatoolkit 10.1, et `sh make.sh` dans `lib/fpn` pour compiler l'IoU. **Compilation d'extension : oui**. Sur Mac, il faudrait réécrire ou remplacer cette partie (non testé).
- Dernière activité : non vérifié (44 commits, pas de date affichée).
- Dépôts miroirs vus (même README, peu visibles) : https://github.com/Moobbot/RelTR, https://github.com/solomonkimunyu/RelTR.

**2. OpenPSG — Open-set Panoptic Scene Graph Generation via Large Multimodal Models (ECCV 2024)**
- Article : https://arxiv.org/pdf/2407.11213 ; code : https://github.com/franciszzj/OpenPSG.
- Poids : **aucun checkpoint trouvé**. Le dépôt publie les fichiers d'annotation `psg_train.json` / `psg_val.json`, pas de poids (vu sur la page du dépôt, sans date). Le dépôt d'origine (Jingkang50, ECCV 2022) a une démo Hugging Face : https://huggingface.co/spaces/ECCV2022/PSG (démo, pas un checkpoint documenté).
- Environnement : utilise mmdet / openseed, entraînement avec `torch.distributed.launch` sur 8 GPU. **Non adapté au Mac** sans effort important.
- Licence, paramètres, nombre de prédicats : non vérifié.

**3. OvSGTR — Fully open-vocabulary scene graph generation (mai 2025)**
- Dépôt : https://github.com/jhelsby/OvSGTR (licence **Apache-2.0**, vue sur la page du dépôt).
- Poids : Google Drive (checkpoint fine-tuné Action Genome, à placer dans `exps/ag_sgdet_ov/checkpoint.pth`) ; pré-entraînement Swin-T `vg-pretrain-coco-swint.pth` sur https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-pretrain-coco-swint.pth ; autres variantes sur le même dépôt HF. Tailles non vérifiées.
- Environnement : PyTorch 1.9.1 + CUDA 11.1 (`torch 1.9.1+cu111`), compilation de GroundingDINO (`python3 setup.py install`). **Difficile sur Mac**.
- Paramètres, nombre de catégories : non vérifié.

**4. OvDSGG — https://huggingface.co/jhelsby/OvDSGG**
- Licence : **apache-2.0** (vue sur la page du modèle).
- Données : Action Genome (graphes dynamiques, vidéo). Vocabulaire ouvert : 25 objets et 18 prédicats utilisés à l'entraînement, sur 36 objets et 26 prédicats au total ; split codé en dur dans `datasets/ag.py`.
- Poids : `exps/{spatial,temporal}_sgdet_ovdsgg_*/checkpoint.pth` (tailles non vérifiées).
- Environnement : Python 3.9 imposé, `./install.sh` en environnement conda. Compilation CUDA non confirmée.
- **Pertinence limitée** : vidéo et Action Genome (scènes du quotidien), pas des peintures.

**5. Modèles Video-LLaVA SGG (Aditya02)** : https://huggingface.co/Aditya02/VideoLlava-SceneGraphGeneration (màj 13/02/2025) et …-Vidor (màj 15/02/2025). Vidéo, pas d'image fixe ; non retenu sans vérification supplémentaire.

**Non trouvé** : aucun modèle PSG ou Visual Genome « Pix2Grp » vérifié lors de la recherche ; la recherche de poids sur la liste de modèles Hugging Face a donné seulement les deux modèles ci-dessus (la page de recherche HF n'a pas été lue en entier).

## Applications à des peintures ou estampes

Aucune étude vérifiée qui applique un de ces modèles à des peintures ou des estampes. Pistes trouvées, à vérifier :
- **CompoVista** : système de graphe de scène pour des collections de peinture chinoise traditionnelle, avec tests utilisateurs auprès d'historiens de l'art. Trouvé via https://www.emergentmind.com/papers/2607.07105 ; identifiant arXiv non vérifié, à confirmer.
- Une revue ETRI (Corée) sur la narration visuelle de tableaux par graphes de scène, qui réclame une grande base d'images de patrimoine peint : https://ksp.etri.re.kr/ksp/article/read?id=61226 (lecture partielle, non vérifiée).
- Un article CEUR-WS (ISWC 2023) sur le regard et le patrimoine, mais c'est de la détection d'objets, pas de la génération de graphe : https://ceur-ws.org/Vol-3632/ISWC2023_paper_429.pdf (non ouvert).

Pas de résultat sur l'application de RelTR, OpenPSG ou OvSGTR à des œuvres d'art. Le problème principal est le décalage de domaine (photos → peintures) : non traité dans les sources trouvées.

## Constats principaux

1. **RelTR** est le seul modèle avec poids publics clairs et nombre de paramètres vérifié (63,7 M), entraîné sur VG avec **50 prédicats fermés** ; licence à confirmer et compilation CUDA à contourner pour un Mac.
2. **OpenPSG** n'a pas de poids publics trouvés : pas utilisable en l'état.
3. **OvSGTR** (Apache-2.0, vocabulaire ouvert) est l'option la plus souple, mais l'environnement (PyTorch 1.9 + CUDA 11.1 + GroundingDINO) est lourd pour un Mac. Aucune étude artistique trouvée.

## Sources

- RelTR : https://github.com/yrcong/RelTR ; https://arxiv.org/html/2201.11460 (paramètres, 50 prédicats, FPS) ; https://arxiv.org/abs/2201.11460 (page de l'article)
- OpenPSG : https://github.com/franciszzj/OpenPSG ; https://arxiv.org/pdf/2407.11213 ; https://huggingface.co/spaces/ECCV2022/PSG
- OvSGTR : https://github.com/jhelsby/OvSGTR ; https://huggingface.co/JosephZ/OvSGTR
- OvDSGG : https://huggingface.co/jhelsby/OvDSGG
- Video-LLaVA SGG : https://huggingface.co/Aditya02/VideoLlava-SceneGraphGeneration
- Art : https://www.emergentmind.com/papers/2607.07105 ; https://ksp.etri.re.kr/ksp/article/read?id=61226 ; https://ceur-ws.org/Vol-3632/ISWC2023_paper_429.pdf
