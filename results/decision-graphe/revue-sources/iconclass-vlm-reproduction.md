# Reproduction du H-F1 Iconclass (Qwen3.5 4B 45,3 / 9B 53,0) : ce qu'on a trouvé

Date de la recherche : 2026-10-25. Lecture seule, rien de téléchargé en entier (poids et jeux de données non téléchargés ; seuls les 2,5 premiers Ko de quelques JSON ont été lus par requête Range). User-Agent : "caypollard/0.1 (research prototype)".

## Résultat principal (à lire d'abord)

1. **Le script `eval_sft.py` et la fonction `_calculate_hierarchical_f1` sont INTROUVABLES publiquement.** Les cartes disent qu'ils vivent dans un dépôt « model-training/iconclass-qwen35 » (avec `train_sft_brill.py`, `eval_corrected.py`, `fuse_rank.py`, `RESEARCH_LOG.md`, `CLAUDE.md`...). Ce dépôt n'existe pas sur le Hub sous aucun type (models/datasets/spaces : HTTP 401 = absent ou privé) ni sur GitHub (davanstrien : aucun dépôt contenant iconclass ; recherche de code GitHub impossible sans authentification ; recherche web : rien). C'est vraisemblablement un dépôt local de l'auteur. **Je n'ai donc pas le texte de la fonction.** Aucun code de H-F1 n'a pu être enregistré ici.
2. **Les 45,3 / 53,0 publiés ne correspondent pas aux mêmes réglages que les fichiers de prédictions publics.** Voir section 2 : le même modèle 4B donne 52,2 de H-F1 glouton dans un fichier de résultats public (max_new_tokens 768, repetition_penalty 1,1) contre 45,3 annoncé. Donc le réglage de décodage change beaucoup le chiffre ; le 53,0 du 9B est donné avec max_new_tokens=384 et sans pénalité (14 % de JSON invalides).

## 1. Script d'évaluation et définition du H-F1

Ce qui est établi (tout indirect, tiré des cartes) :
- Carte 9B : « hierarchical-F1 with partial credit (`eval_sft.py:_calculate_hierarchical_f1` — ancestor matches earn graded credit) ». Donc crédit partiel gradué pour les ancêtres. Le « gradué » n'est pas défini (poids par profondeur ? inconnu).
- Carte du jeu de données : consommateurs `train_sft.py` / `eval_sft.py` lisent la colonne `messages` ; le tour assistant est `{"iconclass-codes": [...]}`.
- Fichiers de sortie de l'auteur (mêmes métriques, produits par un script d'évaluation apparenté) : champs `exact_match_accuracy`, `hierarchical_f1`, `code_precision`, `code_recall`, `code_f1`, `valid_json_pct`, `valid_code_pct`, et par image `hierarchical_f1`, `code_precision`, `code_recall`, `valid_json`, `valid_code_count`, `total_predicted`. Cela permet de **valider une réimplémentation image par image** : l'enregistrement donne `gt_codes`, `predicted_codes`, `generated_text` et `hierarchical_f1` pour chacune des 788 images.
  - Exemple (index 1, `preds_multitask_open.json`) : gt = [31A235, 31A24(+1), 31D15:61BB, 61B(+53), 61B:31A2212(+1)], préd = [31A24(+1), 31D15:61BB, 61B(+53), 61B:31A2212(+1)] -> H-F1 0.9754, précision 1.0, rappel 0.8.
  - Exemple (index 0) : gt = [25F3, 25F38(PHEASANT), 25H, 25H11, 25H213, 34B232(+51)], préd = [25H, 43C11(+4241), 46C131] -> H-F1 0.3946, P 0.333, R 0.167. Avec préd = [25H, 46C131] -> 0.4915 ; préd = [25H, 41A33] -> 0.4915 (fichier sft4b_1ep). Préd = [25G3] -> 0.3556 ; [25H15] -> 0.5787 (rappel de code 0 mais crédit d'ancêtre). Ces points servent de tests de calage.
- Moyenne : les champs par image existent ; le résumé semble être la moyenne des scores par image (hypothèse, non vérifiée sans le code).
- Convention des codes (carte du jeu de données, qui fait foi) : codes complets avec opérateurs `(+n)`, combinaisons `:` (clé) et qualificatifs `(TEXTE)` conservés tels quels ; jetons vides/sentinelles retirés ; ~5 codes texte libre malformés écartés ; plafond de 20 codes/image (1446 images ont la queue coupée).
- Le Space de l'auteur (app.py) utilise la bibliothèque `iconclass` (`from iconclass import init, split_on_colon`) et un `_base(code)` = `code.split("(")[0]` pour décoder. Ce n'est pas l'évaluateur, mais montre la manière de traiter `:` et `(`. URL : https://huggingface.co/spaces/davanstrien/iconclass-9b-zerogpu-test/raw/main/app.py

**Non trouvé** : règle exacte de crédit des ancêtres, analyse de `:` / `(+n)` / `(TEXTE)` dans l'évaluateur, moyenne par image ou globale, traitement des sorties vides/invalides (la carte dit seulement que les JSON invalides comptent comme perte : « 53.0 is a conservative floor », donc une sortie invalide semble recevoir 0 ou un score bas, sans doute inclus dans la moyenne sur 788), gestion des doublons.

Piste de reconstitution possible : prendre `gt_codes` / `predicted_codes` / `hierarchical_f1` des JSON publics (section 5) et chercher la formule qui reproduit les valeurs image par image.

## 2. Réglages d'inférence

Carte 9B (révision 4d83102a), exemple d'usage et section Évaluation :
- Instruction exacte : `"Classify this image using Iconclass codes. Return a JSON object with key 'iconclass-codes' containing a list of codes."` (identique dans app.py du Space).
- Message : `[{"role":"user","content":[{"type":"image"},{"type":"text","text":INSTRUCTION}]}]`, `processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)`.
- `model.generate(**inputs, max_new_tokens=384, do_sample=False)` : glouton, **pas de beam, pas de pénalité de répétition**.
- dtype : `torch_dtype=torch.bfloat16`, `device_map="auto"`.
- « Valid-JSON ≈ 86 % at max_new_tokens=384 » : donc 14 % sorties tronquées/malformées ; 53,0 = plancher.
- Image : l'exemple ouvre l'image en RGB sans redimensionnement ; `processor_config.json` (Qwen2VLImageProcessorFast, patch 16, merge 2) donne `size.shortest_edge=65536`, `longest_edge=16777216` pixels. Donc aucune limite de pixels propre à l'auteur : budget min 65 536 px, max 16,7 M px. À reproduire tel quel.
- Template : `chat_template.jinja` (6833 octets) dans chaque dépôt ; la carte 9B dit « chat template cloned from Qwen/Qwen3.5-4B, `<think>` stripped ».
- generation_config.json : eos 248046, pad 248055, use_cache true, transformers 5.2.0.

Carte 4B (révision 597a5f5c) : AUCUN réglage d'inférence indiqué (ni prompt, ni max_new_tokens, ni décodage). Seulement : « SFT, 1 epoch », base `unsloth/Qwen3.5-4B-Base`, « H-F1 45.3 / hier-recall 46.4 / code-recall 25.6 » sur les 788 images. À noter : la carte 9B cite le 4B à 45.2 (et le fichier README 4B dit 45.3) ; incohérence minime.

Contre-vérification (fichier public de l'auteur, https://huggingface.co/datasets/davanstrien/iconclass-eval-preds/resolve/main/passk/sft4b_1ep.json, 2500 premiers octets) : modèle `davanstrien/qwen35-4b-iconclass-sft-brillfull`, 788 images, `greedy_rep_penalty` 1.1, `max_new_tokens` 768, `thinking_off` false, prompt « native (training prompt) » -> `greedy_hf1` **0.5222**, `greedy_valid_json` 0.9962, `greedy_code_recall` 0.2243. Soit 52,2 au lieu de 45,3. Le rappel de code (22,4 vs 25,6) est même plus bas, donc ce n'est ni la même métrique ni les mêmes réglages. Possibilités (non vérifiées) : le 45,3 vient d'un autre script avec max_new_tokens=384 et sans pénalité ; ou le champ `hierarchical_f1` du JSON passk est calculé autrement. Les autres fichiers (9B multitask 2 ep : hierarchical_f1 0.5873, max 768, rep. pen. d'après app.py) sont de modèles différents (`sft-multitask-2ep`), pas notre 9B 1 epoch.
- Le Space app.py affirme : « Inference recipe (empirically de-risked): enable_thinking=False, max_new_tokens=768, do_sample=False, repetition_penalty=1.1 … without it greedy decoding degenerates ». C'est la recette du modèle multitask 2 ep, pas des cartes brillfull.

**Non trouvé** : journaux d'évaluation bruts de 45,3 et 53,0, fichier de résultats portant exactement ces valeurs pour les révisions épinglées (le fichier passk `sft9b_1ep.json` est probablement le 9B ; non ouvert).

## 3. Lignée et contamination

- 4B : base `unsloth/Qwen3.5-4B-Base` (config.json : `model_name` idem, `unsloth_version` 2026.5.10, architecture `Qwen3_5ForConditionalGeneration`), SFT 1 epoch sur `davanstrien/iconclass-vlm-brillfull`, eval_loss 0,47. Commits : 2026-06-05, dernière = « Document research findings » (597a5f5c).
- 9B : base `unsloth/Qwen3.5-9B-Base`, LoRA r=16 sur toutes couches vision+langage, 1 epoch, bf16, lot 16 x accumulation 2, lr 2e-4 cosine, config `sft` du même jeu (« ~82k train »), eval_loss 0,428, HF Jobs A100-80 Go ~3,6 h. Script cité `train_sft_brill.py --base-model unsloth/Qwen3.5-9B-Base` (introuvable). Dépendances annoncées : `transformers==5.2.0`, `unsloth<2026.6`, `causal-conv1d`, `flash-linear-attention`.
- Jeu de données (révision 8045366f) : train 86 216, test 788, séparés par hachage déterministe du nom de fichier Brill, donc disjoints ; test « propre » pour les modèles entraînés sur ce train. Les points de contrôle plus anciens entraînés sur `iconclass-vlm-sft` (autre découpage) sont **contaminés** sur ce test (avertissement du README). Incohérence mineure : la carte 9B dit « ~82k train » alors que le jeu en compte 86 216 (config `sft` = même train d'après le README du jeu ; à vérifier).
- Conclusion : d'après les cartes, les deux checkpoints brillfull n'ont pas vu les 788 images de test. Vérification indépendante non faite (il faudrait comparer les hachages de noms de fichier ; hors de portée sans télécharger).

## 4. Tailles et faisabilité T4

Tailles (API arbre, https://huggingface.co/api/models/davanstrien/qwen35-4b-iconclass-sft-brillfull/tree/597a5f5c756c21f372bf73d4fa5abcfb745daecb et idem 9B/4d83102a) :
- 4B : 2 fichiers safetensors, 4 972 947 968 + 4 105 672 464 = 9 078 620 432 octets (~9,08 Go, bf16). Tokenizer 19,99 Mo.
- 9B : 4 fichiers : 4 942 706 120 + 4 987 757 928 + 4 954 810 240 + 3 934 446 832 = 18 819 721 120 octets (~18,8 Go, bf16).
- Jeu de données : test = 59 222 740 octets (1 fichier parquet) ; train = 11 parquets d'environ 600 Mo chacun.

T4 16 Go :
- 4B : ~9,1 Go de poids, tient en fp16 avec de la marge pour une image + 384 jetons. Mais la T4 n'a pas de bf16 matériel ; il faudrait charger en float16 (poids bf16 -> fp16, petit risque de débordement dans les couches de linear-attention ; à tester contre bf16 sur quelques images). Le résultat publié est en bf16, donc un écart numérique est possible.
- 9B : ~18,8 Go, **ne tient pas** en fp16 sur une T4 de 16 Go. Il faut quantification 8 bits/4 bits (change les chiffres), ou GPU plus gros (L4 24 Go, A10G, A100), ou 2 x T4 avec `device_map="auto"`.
- Support : architecture `qwen3_5` (couches `linear_attention` type Gated DeltaNet + `full_attention` tous les 4). La carte 9B dit « transformers==5.2.0 (only version with qwen3_5) » ; le generation_config.json indique transformers 5.2.0. Pour la vitesse, il faut `flash-linear-attention` et `causal-conv1d` ; sans eux le repli PyTorch est très lent (non vérifié). Non testé sur T4.

## 5. Fichiers de prédictions publics utiles au calage

Dépôt https://huggingface.co/datasets/davanstrien/iconclass-eval-preds (arbre : https://huggingface.co/api/datasets/davanstrien/iconclass-eval-preds/tree/main?recursive=true) : `passk/sft4b_1ep.json` (4B brillfull), `passk/sft9b_1ep.json` (probablement le 9B brillfull), `passk/sft9b_2ep.json`, `passk/sft4b_orig.json` (modèle sur jeu tronqué), `passk/base4b_2shot.json`, `passk/base9b_2shot.json`, `passk/grpo4b_v3.json`, `preds_multitask_open.json`, `preds_multitask_completion.json`, `preds_brill_n_2ep_open.json`, `preds_brill_n_2ep_oracle.json`, `internvl35_4b_sft_brillfull_1ep_test788.json` (InternVL3.5, max 768, rep. pen. 1,1, H-F1 0.5135), `judgebench/judge_bench.json`. Ces fichiers donnent par image les codes vérité, les prédictions et le H-F1 : c'est la meilleure source pour retrouver la formule.
Autres pages ouvertes : https://huggingface.co/davanstrien/qwen35-4b-iconclass-sft-brillfull/raw/597a5f5c756c21f372bf73d4fa5abcfb745daecb/README.md, .../config.json, .../processor_config.json, .../generation_config.json ; https://huggingface.co/davanstrien/qwen35-9b-iconclass-sft-brillfull/raw/4d83102aec672c09eacb07568c8f81367616a1d7/README.md ; https://huggingface.co/datasets/davanstrien/iconclass-vlm-brillfull/raw/8045366ff092a06e8e1c4e47f1eb43cb2a54d732/README.md ; arbre du jeu de données ; recherches API Hub (davanstrien, uv-scripts, « eval_sft », « hierarchical_f1 »), README de 81 dépôts iconclass de davanstrien (aucun lien vers eval_sft.py), https://api.github.com/users/davanstrien/repos (rien d'iconclass), https://huggingface.co/datasets/davanstrien/iconclass-predictions (+ iconclass-sft-predictions, iconclass-vlm-sft, iconclass-anchor-preds : ni script ni métrique).

## Ce qu'il faut faire ensuite
1. Demander le dépôt `model-training/iconclass-qwen35` à l'auteur (Daniel van Strien, via une issue/discussion publique du Hub, sans e-mail), ou reconstruire le H-F1 en l'ajustant aux valeurs par image de `passk/sft4b_1ep.json`.
2. Lancer le 4B en bf16/fp16 avec max_new_tokens=384, glouton, sans pénalité, enable_thinking=False, puis aussi 768 + pénalité 1,1, et comparer à 45,3 et 52,2.
3. Le 9B exige un GPU d'au moins ~24 Go.
