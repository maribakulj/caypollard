# Construire un générateur de graphes de scène avec peu de données annotées (état au 17/10/2026)

Méthode : ~20 appels web ; seules les pages réellement lues sont citées. Tout ce qui n'a pas été lu sur une page primaire est marqué « non vérifié ». Les extractions de WebFetch passent par un petit modèle : chiffres à recouper avant publication.

## 1. SGG open-vocabulary / open-set (OvSGTR, OpenPSG, PGSG, SGTR, RelTR, EGTR, 2025-2026)

### Takeaway
OvSGTR (Apache-2.0, poids publics, backbones Swin-T/B de GroundingDINO) et R1-SGG (Apache-2.0, poids 7B publics) sont les deux briques ouvertes les mieux documentées. OpenPSG a du code mais pas de poids vérifiés. Rien n'a été vérifié pour PGSG/Pix2Grp, SGTR, RelTR, EGTR.

### Cited Findings
- OvSGTR : transformeur de bout en bout, alignement visuel-concept pour nœuds et arêtes, pré-entraînement relationnel sur paires image-légende, distillation de connaissances ; soumis 18/11/2023, v2 7/10/2024 ; évalué sur Visual Genome — [arXiv 2311.10988](https://arxiv.org/abs/2311.10988)
- OvSGTR code sous Apache-2.0 ; backbones Swin-T et Swin-B (GroundingDINO) ; points de contrôle sur Hugging Face (JosephZ/OvSGTR), dont variantes pré-entraînées « MegaSG » — [GitHub gpt4vision/OvSGTR](https://github.com/gpt4vision/OvSGTR)
- OvSGTR, R@50 sur classes nouvelles (README) : OvD-SGG 12,06 (Swin-T) / 15,58 (Swin-B) ; OvR-SGG 13,47 / 16,37, jusqu'à 22,86 (Swin-B MegaSG) ; OvD+R : objet nouveau 14,32 / 17,58, relation nouvelle 9,19 / 14,62. Le README ne donne pas mR@50 ni le matériel GPU (environnement PyTorch 1.9.1 / CUDA 11.1, ancien) — [GitHub gpt4vision/OvSGTR](https://github.com/gpt4vision/OvSGTR)
- OpenPSG (ECCV 2024, Zhou, Zhu, Caesar, Shi) : prédiction de relations en ensemble ouvert avec grands modèles multimodaux, segmenteur panoptique ouvert, « relation query transformer » filtrant les paires ; se dit premier à proposer la tâche — [arXiv 2407.11213](https://arxiv.org/pdf/2407.11213) ; code [github.com/franciszzj/OpenPSG](https://github.com/franciszzj/OpenPSG)
- Dépôt OpenPSG : dépend d'OpenSeeD, aucun poids pré-entraîné listé (seulement les JSON PSG), licence non visible sur la page, lancement d'entraînement à 8 processus — [GitHub OpenPSG](https://github.com/franciszzj/OpenPSG)
- R1-SGG « Compile Scene Graphs with RL » (Chen et al., 18/04/2025, rév. 26/05/2025) : MLLM générant objets + triplets, SFT puis RL (récompenses centrées graphe + format) ; évalué VG150 et PSG ; réduit fortement le taux d'échec — [arXiv 2504.13617](https://arxiv.org/abs/2504.13617)
- R1-SGG : Apache-2.0 ; poids R1-SGG-7B et R1-SGG-Zero-7B sur HF ; formation sur VG150 ; SFT ≈ 4 h (2B) / 10 h (7B) sur 4×A100 ; GRPO ≈ 12 h (2B) sur 16×A100 ; 7B sur cartes 24 Go possible avec Zero3 mais lent — [GitHub gpt4vision/R1-SGG](https://github.com/gpt4vision/R1-SGG)
- SceneGraphVLM (13/05/2026) : petits VLM, graphe sérialisé en format TOON, SFT puis RL avec récompenses anti-hallucination, évalué PSG/PVSG/Action Genome, ~1 s par graphe ; le modèle de base et la taille ne sont pas précisés sur la page ; code github.com/markus0440/SceneGraphVLM — [arXiv 2605.13667](https://arxiv.org/abs/2605.13667)
- VOST-SGG (déc. 2025) : LoRA sur Qwen2.5-VL-7B-Instruct, ≈20 M paramètres entraînés sur 7,6 G, une époque sur Action Genome (extrait de recherche, page non ouverte) — [arXiv 2512.05524](https://arxiv.org/pdf/2512.05524)

### Inferences
- Pour un petit groupe, la voie la plus économique est : OvSGTR (détecteur ouvert à fine-tuner) ou un VLM 2-7B fine-tuné au format texte (R1-SGG montre que c'est faisable, mais avec 4×A100 ; en LoRA sur 16 Go il faudra du 2-3B).
- Les chiffres OvSGTR sont sur photos VG ; aucun transfert à l'art n'est documenté.

### Gaps
- Non vérifié : PGSG / Pix2Grp, SGTR, RelTR, EGTR (licences, poids, chiffres) ; mR@50 d'OvSGTR et d'OpenPSG ; licence d'OpenPSG ; nombre de paramètres d'OvSGTR ; modèles SGG 2026 hors SceneGraphVLM (aucune recherche systématique 2026 faite).

## 2. Supervision faible par légendes

### Takeaway
Le gain de LLM4SGG est net en mR@K mais la supervision faible reste très loin de la supervision complète en R@K (≈9-10 contre ≈32 sur VG, SGDet).

### Cited Findings
- LLM4SGG (Kim et al., CVPR 2024, pp. 28306-28316) : un LLM (ChatGPT, version non précisée) extrait les triplets des légendes et aligne entités/prédicats ; corrige la sur-simplification sémantique et la faible densité de graphe ; arXiv 16/10/2023, v8 29/07/2024 — [arXiv 2310.10404](https://arxiv.org/html/2310.10404v8) ; [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_LLM4SGG_Large_Language_Models_for_Weakly_Supervised_Scene_Graph_Generation_CVPR_2024_paper.html)
- Table 2 (VG, SGDet, R@50 / mR@50) : Motif supervisé 31,89 / 6,38 ; SGNLS (ICCV 2021) 3,80 / 2,51 ; SGNLS+LLM4SGG (COCO) 5,09 / 4,08 ; VS³ (CVPR 2023) 6,69 / 1,73 ; VS³+LLM4SGG (COCO) 8,91 / 7,11 ; avec légendes CC 9,47 / 5,40 ; avec légendes VG 18,40 / 6,26. Images : COCO 64 K, CC 145 K, VG 57 K — [arXiv 2310.10404v8](https://arxiv.org/html/2310.10404v8) (lecture partielle, 100 000 caractères sur 147 373)
- PRISM-0 (Elskhawy et al., 1/04/2025, rév. 15/11/2025) : pipeline zéro-shot détection + VLM + LLM + validation VQA ; « à égalité » avec le faible supervisé de l'état de l'art selon l'abstract, pas de chiffres sur la page ; pas de code lié — [arXiv 2504.00844](https://arxiv.org/abs/2504.00844)
- Open World SGG (Dutta et al., 2025, atelier CVPR) : cadre sans entraînement avec VLM, évalué VG, Open Images V6, PSG (via résumé de recherche, PDF illisible) — [arXiv 2506.08189](https://arxiv.org/pdf/2506.08189)

### Inferences
- Les méthodes par légendes n'ont jamais été testées sur de l'art dans ce qui a été lu ; les légendes d'œuvres (musées) pourraient jouer ce rôle, hypothèse non testée.

### Gaps
- Non vérifié : Zhong et al. ICCV 2021 (SGNLS) et Ye & Kovashka CVPR 2021 (LSWS) : seuls leurs chiffres de seconde main via le tableau LLM4SGG sont cités (LSWS 3,29 R@50, SGNLS 3,80) ; papiers originaux non ouverts. Travaux faibles post-2024 non explorés.

## 3. VLM comme générateurs de graphes, benchmarks, distillation

### Takeaway
Les VLM généralistes sont mauvais en zéro-shot sur VG150 (format, taux d'échec, vocabulaire) ; le SFT (+RL) d'un modèle 2-7B corrige l'essentiel. Aucune recette « maître VLM → élève LoRA » validée de bout en bout n'a été trouvée.

### Cited Findings
- R1-SGG compare GPT-4o, Gemini 1.5 Flash, LLaVA v1.5, Qwen2-VL sur VG150 et rapporte une performance zéro-shot faible (résumé de recherche, chiffres non lus) — [arXiv 2504.13617](https://arxiv.org/abs/2504.13617)
- Neau et al. 2025 : les benchmarks SGG ont un vocabulaire trop petit ; métrique sans référence pour évaluer les VLM sur les relations — [arXiv 2509.01209](https://papers.cool/arxiv/2509.01209) (non ouvert, via résumé)
- SceneAlign (janv. 2026) existe sur l'alignement raisonnement/graphes (titre seul) — [arXiv 2601.05600](https://arxiv.org/pdf/2601.05600)

### Inferences
- Pipeline plausible : annoter ~quelques centaines d'œuvres avec un VLM fort, corriger à la main, LoRA sur Qwen2.5/3-VL 2-3B ; coûts API non chiffrés ici.

### Gaps
- Aucun chiffre vérifié sur GPT-4o/Gemini/Claude en SGG ; coûts par image ; distillation maître→élève documentée ; benchmarks 2026.

## 4. Transfert photo → art / dessin / BD

### Takeaway
Le fossé de domaine est documenté de longue date ; Human-Art fournit 50 k images annotées pour combler, mais les chiffres ViTPose 0,555→0,759 n'ont pas pu être retrouvés.

### Cited Findings
- Human-Art (CVPR 2023, Ju, Zeng, Wang, Xu, Zhang) : ≈50 k images, >123 k personnes, 5 scénarios naturels + 15 artificiels (sculptures, peintures, dessins animés), boîtes, points clés, points de contact, texte, 2D/3D ; arXiv 5/03/2023 — [arXiv 2303.02760](https://arxiv.org/abs/2303.02760) ; [CVF PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Ju_Human-Art_A_Versatile_Human-Centric_Dataset_Bridging_Natural_and_Artificial_Scenes_CVPR_2023_paper.pdf)
- PeopleArt : détection de personnes, CNN affiné <60 % AP (Westlake 2016) — [arXiv 1610.08871](https://arxiv.org/pdf/1610.08871)
- IconArt (Gonthier 2018) : détection faiblement supervisée avec étiquettes au niveau image seulement — [ECCVW 2018](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Gonthier_Weakly_Supervised_Object_Detection_in_Artworks_ECCVW_2018_paper.pdf)
- DEArt : jeu de données d'art européen pour détection d'objets — [arXiv 2211.01226](https://arxiv.org/pdf/2211.01226) (statistiques non lues)
- Cross-depiction (Crowley & Zisserman 2015) : aucune méthode de l'époque ne le résout — [arXiv 1505.00110](https://ar5iv.arxiv.org/html/1505.00110)

### Inferences
- Pour la pose d'œuvres, fine-tuner sur Human-Art (licence à vérifier) est la voie la moins chère.

### Gaps
- Non vérifié : ViTPose 0,555→0,759 ; nombre d'étiquettes cibles nécessaires ; licence Human-Art ; jeux de relations pour manga/BD ; premier lien ar5iv/arxiv 2102.06529 (style transfer) non lu. Mon essai d'ouvrir arXiv 2207.02498 comme Human-Art était une erreur d'identifiant (autre article) : ignoré.

## 5. Petits VLM ouverts pour sortie structurée, fine-tuning 16 Go

### Takeaway
Qwen2.5-VL/Qwen3-VL (localisation + JSON), Florence-2 (MIT, 0,77 B) et PaliGemma 2 (licence Gemma) sont des bases crédibles ; Florence-2 est le plus léger et le plus libre mais produit boîtes/légendes, pas des graphes.

### Cited Findings
- Qwen2.5-VL-3B-Instruct : famille 3/7/72 B ; localisation par boîtes/points, « sorties JSON stables » (affirmation de la fiche) ; baisser `max_pixels` réduit la mémoire ; licence non lue sur la page (non vérifié) — [HF Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Qwen3-VL : tailles denses de 2 B à 32 B plus MoE 30B-A3B et 235B-A22B ; Apache-2.0 selon agrégateurs (pas la fiche officielle) ; 32B Instruct le 22/10/2025 — [DataLearner](https://www.datalearner.com/en/ai-models/pretrained-models/qwen3-vl-32b-instruct) ; [Simon Willison 23/09/2025](https://simonwillison.net/2025/Sep/23/qwen3-vl)
- Florence-2-large : MIT, ≈0,77 B, grounding de légende, détection, légendes de régions denses ; article 10/11/2023 — [HF microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- PaliGemma 2 3B (pt-448) : licence Gemma (acceptation requise + politique d'usage), 3/10/28 B, détection/segmentation par jetons ; modèle de base à affiner ; code big_vision — [HF google/paligemma2-3b-pt-448](https://huggingface.co/google/paligemma2-3b-pt-448)
- LoRA sur Qwen2.5-VL-7B : ≈20 M de paramètres entraînés — [arXiv 2512.05524](https://arxiv.org/pdf/2512.05524)
- R1-SGG : 7B sur cartes 24 Go possible seulement lentement — [GitHub R1-SGG](https://github.com/gpt4vision/R1-SGG)

### Inferences
- Sur T4/P100 16 Go : 2-3B en LoRA/QLoRA, images réduites, probablement faisable ; 7B juste (non mesuré). Les P100 n'ont pas bf16 ni flash-attention (connaissance générale, non vérifié ici).

### Gaps
- Non vérifié : InternVL, SmolVLM, Moondream (aucune page ouverte) ; licence exacte Qwen2.5-VL 3B (la licence « Qwen research » concerne le 3B, à confirmer) ; mémoire mesurée sur T4/P100.
