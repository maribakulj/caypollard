# Migration du 25/10/2026

caypollard reste l'atelier : briques, bancs, visionneuse, mesures d'atelier. Les entraînements et les expériences qui produisent des
modèles vivent désormais ailleurs ; caypollard ne dépend d'aucun de ces dépôts.

- **lermolieff** (dépôt public en préparation, pas encore publié) : la tête Iconclass (variantes C-RADIO et PE-Spatial), son code
  d'inférence, d'entraînement et de mesure, ses résultats.
- **porkyevitch** (privé) : les expériences d'entraînement et d'évaluation, copiées à l'identique avec leur provenance (commit
  d'origine, empreintes, version des scripts réellement exécutée).

Tout ce qui est retiré ici reste dans l'historique git : le dernier commit qui contient ces fichiers est `c0bfe6c`
(branche pilote-figures).

| ancien chemin (caypollard) | où |
|---|---|
| `scripts/remote_iconclass_head.py`, `results/iconclass-tete/` | porkyevitch `experiences/2026-10-25-iconclass-tete/` ; code repris dans lermolieff |
| `scripts/remote_iconclass_vlm_bench.py` | porkyevitch `experiences/2026-10-25-iconclass-tete/` |
| `scripts/iconclass_par_usage.py`, `results/iconclass-tete-pe/` | porkyevitch `experiences/2026-10-25-iconclass-tete-pe/` ; résumé dans lermolieff `lermolieff/resultats/` |
| `scripts/remote_xanto_parts.py`, `results/xanto-parties/` | porkyevitch `experiences/2026-10-25-xanto-parties/`, à rendre au projet xanto |
| `results/decision-graphe/`, `scripts/build_scene_train_set.py`, `scripts/remote_scene_qwen.py`, `scripts/colab_train_scene.sh` | porkyevitch `experiences/2026-10-17-graphe-de-scene/` |

Restent ici :
- `scripts/remote_head_scene.py` et `scripts/remote_train_scene_lora.py` : ils sont la source de briques de l'atelier
  (`src/caypollard/atelier/bricks.py`, `results/atelier/`) ; une copie est aussi dans porkyevitch.
- L'ancien banc extérieur Iconclass, tiré d'Emblematica (`scripts/build_external_iconclass_bench.py`, `scripts/remote_external_iconclass.py`,
  `scripts/remote_external_vlm.py`, `results/iconclass-externe/`) : c'est un banc, avec ses listes gelées et signées, intactes,
  gardé pour la reproduction des résultats déjà obtenus. Il ne sert plus à décider les essais suivants. Une copie est dans porkyevitch.
- Les poids (têtes, adaptateur LoRA) ne sont pas dans git : ils sont archivés à part, avec leur empreinte. Les deux têtes seront
  publiées sur Hugging Face avec lermolieff.
