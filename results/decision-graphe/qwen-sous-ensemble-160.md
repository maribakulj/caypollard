# Petits modèles sans entraînement sur les 160 images fixées (17/10/2026)

Banc `relations`, sous-ensemble `scripts/relations_subset160.json` (10 images par sujet, graine 0). Bon sujet en premier (précision moyenne entre parenthèses). Même consigne pour les trois lecteurs (`SCENE_PROMPT`, `src/caypollard/atelier/bricks.py`).

| Lecteur | Mots | Sens (MiniLM) | Figures seules | Graphes vides | s/image |
|---|---|---|---|---|---|
| Opus (`opus-scene`) | 0,731 (0,534) | 0,750 (0,582) | 0,613 | 0 | 15 (appel claude -p) |
| Qwen2.5-VL-7B (`qwen25vl7b-scene`, Colab L4) | 0,181 (0,168) | 0,237 (0,213) | 0,206 | 1 | 21,5 |
| Qwen2.5-VL-3B (`qwen25vl3b-scene`, Kaggle T4) | 0,169 (0,151) | 0,188 (0,167) | 0,125 | 5 | 12,3 |
| SigLIP (image entière) | 0,425 (0,254) | | | | |

Constats sur les graphes : Qwen 7B sort du vocabulaire fermé (rôle « bird », relation « flies near », « child » pour « infant ») et décrit surtout la disposition (« stands beside », « is above ») ; Qwen 3B répète des relations génériques (« man carries man »). Coût Colab : environ 2 unités pour le 7B (L4, 1,54 unité par heure).
