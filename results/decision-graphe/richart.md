# RichArt — faits vérifiés (tâche 1)

Dépôt : https://huggingface.co/datasets/MihaiBogdanBindila/RichArt — licence déclarée `cc-by-nc-4.0` (README et métadonnées, vérifié sur https://huggingface.co/api/datasets/MihaiBogdanBindila/RichArt et https://huggingface.co/datasets/MihaiBogdanBindila/RichArt/raw/main/README.md).
Fichiers téléchargés (annotations seulement, aucune image) dans `raw/` :
- `raw/train.json` 18,6 Mo, `raw/val.json` 2,3 Mo, `raw/test.json` 2,3 Mo, `raw/README.md`.
- Tailles vérifiées avant téléchargement (en-tête `content-length`) : https://huggingface.co/datasets/MihaiBogdanBindila/RichArt/resolve/main/annotations/train.json (18 630 636 o), …/val.json (2 304 825 o), …/test.json (2 317 562 o). Tous sous 200 Mo.

## a) Format exact

Un fichier = un tableau JSON (pas du JSONL). Une entrée = **un tableau** (une peinture), pas une région. Chaque peinture contient des métadonnées et un dictionnaire `objects` : une **entité symbolique** (clé = nom court, ex. `"zephyr"`) avec une **description** et une **liste de boîtes**. Une entité peut donc avoir plusieurs boîtes (`bounding_boxes`), chacune `[score_GroundingDINO, [x1, y1, x2, y2]]`.

Exemple réel (`annotations/val.json`, id 0, « Flora and Zephyr », Jacopo Amigoni) :

```
"objects": {"zephyr": {"description": "Zephyr is the god of wind.",
  "bounding_boxes": [[0.4967..., [65.70, 410.84, 474.95, 1027.86]]]}}
```

Réponse à la question : ce n'est **pas** « une région = une boîte + une description ». C'est « une entité = une description + 1 à N boîtes ». Chiffres vérifiés en parcourant les trois fichiers : 7 087 peintures, 12 153 entités, 20 882 boîtes au total (cohérent avec le README). 8 835 entités ont 1 boîte, 1 656 en ont 2, le reste davantage (jusqu'à 10 et plus).

## b) Relations entre entités (échantillon de 300, graine 0)

Méthode : `random.seed(0)` puis `random.sample(liste_des_12153_entités, 300)` (ordre des trois fichiers). Fichier : `sample300.json`. Classement fait **à la main par une seule lecture** (moi), pas de double annotation : la proportion est donc approximative (marge d'erreur estimée ±5 points à cause du jugement, notamment pour les « relations faibles » comme « X est dans Y »).

- **86 / 300 (29 %)** décrivent une relation explicite entre l'entité et une AUTRE entité (verbe ou préposition d'interaction : tient, regarde, s'agenouille devant, embrasse, présente, sert de contenant, derrière, etc.).
- Parmi ces 86, **31** ont leur partenaire **aussi annoté avec une boîte dans la même image** (clé présente dans `objects`, ex. `christ`, `st joseph`, `cupid`, `virgin`). Soit **31 / 300 (10 %)** des descriptions tirées au hasard, et **≈ 36 %** des relations.
- Les 55 autres relations n'ont pas de partenaire boîté : soit l'autre entité est décrite dans le texte mais pas annotée (ex. « Gregory » dans `monk soul`, « Cupid » dans `mirror`), soit elle n'a pas de clé (ex. « Deianira » est annotée, mais « Venus » dans `adonis` ne l'est pas).

Lecture : le jeu fournit une **description relationnelle** mais l'essentiel des relations dans le texte pointe vers des personnages **non ancrés**. Un graphe de scène « entité–entité avec boîtes » ne serait supervisé que sur environ un tiers des relations, et à peu près un dixième des descriptions.

Fichier détaillé : `rel_others.json` (entités relationnelles + autres entités de la peinture) ; échantillon brut : `sample300.json`.

## c) Les 30 mots de relation les plus fréquents (compte simple)

Méthode : liste fermée (prépositions + verbes d'interaction fléchis, ~150 formes) comptée sur les 12 153 descriptions d'entités. **Pas d'étiquetage grammatical** : « of », « in », « to » dominent parce que ce sont des prépositions très fréquentes et souvent non relationnelles (« of » = possession). Script : `verbs.py`.

| rang | mot | occurrences |
|---|---|---|
| 1 | of | 16 242 |
| 2 | in | 9 164 |
| 3 | to | 5 757 |
| 4 | with | 3 934 |
| 5 | on | 3 217 |
| 6 | by | 2 935 |
| 7 | from | 1 699 |
| 8 | at | 1 652 |
| 9 | for | 1 318 |
| 10 | into | 515 |
| 11 | over | 398 |
| 12 | against | 306 |
| 13 | between | 296 |
| 14 | behind | 291 |
| 15 | before | 287 |
| 16 | around | 280 |
| 17 | holding | 257 |
| 18 | through | 245 |
| 19 | holds | 242 |
| 20 | above | 217 |
| 21 | along | 201 |
| 22 | wearing | 192 |
| 23 | stands | 192 |
| 24 | standing | 183 |
| 25 | placed | 167 |
| 26 | under | 161 |
| 27 | wears | 160 |
| 28 | among | 139 |
| 29 | towards | 138 |
| 30 | held | 126 |

Lecture : hors prépositions grammaticales, les verbes de relation les plus fréquents sont « holding / holds / held » (tenir), « wearing / wears » (porter), « stands / standing » (se tenir debout) : le corpus est surtout **statique** (postures, objets tenus ou portés), peu de verbes d'action entre deux personnages.

## d) 10 exemples recopiés mot pour mot

Format : id de peinture (`id` du JSON) — titre — clé de l'entité — description (texte exact).

1. (7771) Braque Family Triptych (central panel) — `globe earth` — « The globe of the earth is held in Christ's left hand while he raises two fingers of his right in blessing. »
2. (7237) Madonna with Sts John the Baptist and Donatus — `virgin` — « Virgin and Child with Sts John the Baptist and Donatus in Pistoia Cathedral. The Virgin and the infant are seated in mid-ground, and whereas the Virgin's legs project towards St Donatus, her face turns to St John the Baptist. »
3. (257) Judith with the Head of Holofernes — `judith` — « The Jewish heroine Judith presents the severed head of the Assyrian general who besieged her city, having seduced and then beheaded him with his own sword. »  (suite de la description non recopiée)
4. (19) Diana and Cupid — `diana` — « Diana holds Cupid's bow out of his reach, an action that is probably intended as a reproof, since Cupid's use of the bow was considered capricious and misdirected (Ovid, Metamorphoses, Book 1, lines 455ff.), and the Diana is inspired by a famous Roman statue in the Vatican. »  (extrait)
5. (2811) Venus, Adonis, and Cupid — `adonis` — « Adonis is shown with hounds that will accompany him on a hunt with a wild boar, which ultimately leads to his demise. »  (extrait)
6. (2374) Madonna Litta — `infant jesus` — « The Madonna Litta is a late 15th-century painting of the Madonna nursing the infant Jesus. »
7. (8627) The First of May 1851 — `queen victoria` — « The aged Duke of Wellington is presenting a casket to his one-year-old godson, Prince Arthur, Duke of Connaught, who is supported by Queen Victoria. »  (extrait)
8. (7654) Diptych of Maarten Nieuwenhove — `convex mirror` — « As in the Arnolfini Marriage, a convex mirror hangs on the wall behind the Virgin and captures the entire scene, and we can see the nobleman at prayer and the Virgin at his side. »
9. (3028) The Mystic Marriage of St Catherine — `st catherine` — « St Catherine, identified as a princess by her crown, turns shyly towards Christ, who places a ring on her finger. »
10. (9013) Tancred Baptizing Clorinda — `christian knight tancred` — « Christian knight Tancred (d. 1112 in Antioch) is depicted baptizing his mortally wounded opponent at that opponent's own wish, after a nocturnal duel. »

Note : les exemples 3, 4, 5, 7 sont des extraits (coupés avec « … »-style, sans points de suspension) ; les entrées complètes sont dans `raw/*.json` (champ `description` de l'entité `objects[clé]`).

## Constats principaux

1. Une entité symbolique (description + 1..N boîtes), pas une boîte par région : 12 153 entités, 20 882 boîtes.
2. Environ 29 % des descriptions échantillonnées décrivent une relation vers une autre entité, mais seulement ~10 % ont **les deux bouts** annotés avec une boîte dans l'image.
3. Les relations du corpus sont surtout statiques (tenir, porter, se tenir) ; peu de verbes d'action entre personnages.

## Sources vérifiées

- Dataset et README : https://huggingface.co/datasets/MihaiBogdanBindila/RichArt
- Métadonnées (licence, date de dernière modification 2026-04-15) : https://huggingface.co/api/datasets/MihaiBogdanBindila/RichArt
- Tailles des fichiers : URL `resolve/main/annotations/{train,val,test}.json` ci-dessus.
- Article cité (ICMR 2026, « Fine-Grained Cross-Modal Retrieval in Art via Region-Level Grounding of Symbolic Narratives ») : non vérifié (je n'ai pas ouvert l'article ; seul le README le cite). Dépôt de code cité dans le README : non vérifié (pas ouvert).
