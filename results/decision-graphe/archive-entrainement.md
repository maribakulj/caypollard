# Archive figée pour l'entraînement LoRA (23/10/2026)

Graphes Opus d'entraînement (`data/derived/atelier/entrainement/scene--opus-scene`) contrôlés contre `data/derived/scene-train/manifest.json` : 1 200 attendus, 1 200 présents, 0 absent, 0 JSON invalide (`figures` et `relations` sont des listes), 17 graphes sans figures (exclus de l'entraînement et comptés à part), 0 fichier en trop. Code au commit 9b92c3a.

Empreintes SHA-256 des archives envoyées à Colab :

```
eb290043dd0e771dd327bf34f9a92ac54b552703bd877b97925e08e69a193279  code.tar
116581dbd5f2ef710daa7958191d7b678e74392dd249812c013a3dfa2ae7cded  train.tar.gz
072c2b9e53e3bdf1df490075b635bbfffe762e680f86042fc8a8f059e78deeab  teacher.tar.gz
92028d924a84fbc985d676e74c04403ddad0d1009575906088adab066b0d03f3  relations.tar.gz
5d169e0dac5d1db68ad20cc64c5a4950dd1d281156463d27bd0b33ff905be33f  subjects.tar.gz
```
