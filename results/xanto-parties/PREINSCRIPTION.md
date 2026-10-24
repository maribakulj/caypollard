# Pose et parties de la figure contre le vecteur global : test sur l'oracle Xanto (règles écrites avant calcul)

Date : 25/10/2026, avant tout calcul. Script : `scripts/remote_xanto_parts.py` (dépôt caypollard). Données : celles de l'oracle Xanto,
lues sans modification (`~/Documents/Codex/2026-10-08-continuation-claude/recherche-sources` et `corpus-images/rovigo`).

## Question

Le test oracle de Xanto (`xanto/recherche-sources/RESULTAT_ORACLE_DINOV3_TOP100.md`) montre que, même avec la bonne figure gravée
découpée à la main, le vecteur de classe DINOv3-L de la figure peinte ne la place dans les 10 premières parmi 6 891 découpes que
9 fois sur 49. Une comparaison par la pose, ou partie par partie (tête, buste, bras, jambes), relie-t-elle mieux la figure peinte
à la figure gravée ?

## Protocole (identique à l'oracle DINOv3 sauf la mesure de ressemblance)

- Mêmes 49 requêtes, mêmes découpes oracle, mêmes 6 844 découpes témoins, construites par le code de `oracle_dinov3.py` recopié
  tel quel ; même rang (rang de la découpe oracle, autres découpes oracle de la même figure exclues), même meilleur des deux sens
  de la figure du plat (endroit et miroir). Q08 et Q17 (figure introuvable) comptent comme échecs.
- **Contrôle de reproduction, avant toute variante** : B0 recalculé à partir du cache `cache_oracle_dinov3l_384.npy` doit
  redonner exactement les rangs de `resultats_oracle_dinov3.csv`. Sinon le calcul s'arrête.

## Variantes (toutes fixées ici, aucune ajoutée après coup)

- **B0** : vecteur de classe DINOv3-L de la découpe entière (la référence, 9/49 dans les 10 premières).
- **P, pose** : ViTPose-B entraîné sur Human-Art (brique `vitpose-humanart` de l'atelier) dans la boîte de chaque découpe ;
  articulations gardées au-dessus de 0,3. Les deux squelettes sont centrés (moyenne des articulations communes) et mis à
  l'échelle (écart quadratique moyen à ce centre) ; ressemblance = − distance moyenne entre articulations communes. Moins de 6
  articulations communes : ressemblance la plus basse. Le sens miroir est obtenu en retournant la photo du plat avant la pose.
- **F, fusion B0 + P** : rangs fusionnés par 1/(60 + rang), constante déjà utilisée par Xanto.
- **K, parties** : à partir des articulations, six boîtes (tête : nez et épaules ; buste : épaules et hanches ; bras gauche et
  droit : épaule, coude, poignet ; jambe gauche et droite : hanche, genou, cheville), chacune agrandie de 25 % et d'au moins
  un quart de la hauteur de la figure. Chaque partie est décrite comme B0 (DINOv3-L, carré au gris moyen, 384 px). Ressemblance
  = moyenne des cosinus des parties présentes des deux côtés ; moins de deux parties communes : cosinus de B0.
- **KP, parties + pose** : rangs de K et de P fusionnés par 1/(60 + rang).

## Règle

Une variante **gagne** si elle place la découpe oracle dans les 10 premières pour **au moins 18 requêtes sur 49** (le double de
B0) **et** si, requête par requête contre B0, elle fait monter la découpe oracle plus souvent qu'elle ne la fait descendre avec
p < 0,0125 au test du signe exact (égalités exclues ; seuil de Bonferroni pour quatre variantes).

Publiés sans valeur de verdict : rang 1, 10 et 50 premières, rang médian ; résultats par type (figure, groupe, composition) ;
part des découpes où la pose n'a pas trouvé six articulations ; une planche de squelettes et de parties pour six requêtes tirées
au sort (graine 0), à regarder avant de lire les chiffres.

## Portée

C'est un diagnostic : les 49 requêtes ont déjà servi à plusieurs choix de Xanto. Si une variante gagne, elle justifie de construire
la chaîne par parties ; elle ne sera retenue qu'après un nouveau lot fermé de paires. Les figures du plat sont petites (hauteur
médiane d'environ 230 px) : les parties y font quelques dizaines de pixels, ce qui peut limiter K.

Relecture par Codex (w3:p3) avant l'envoi sur Kaggle.

## Amendements du 25/10/2026, avant tout calcul complet (relecture Codex : `~/Documents/reviews/caypollard-xanto-parties-codex-2026-10-25.md`)

1. **Lot réel : 48 requêtes, pas 49.** Q49 a une découpe oracle (`c_oracle.csv`) mais aucune boîte sur le plat dans
   `requetes_boites.csv` ; elle n'a jamais été évaluée (le fichier publié a 48 lignes, dont Q08 et Q17 sans rang). Dénominateur :
   48 ; Q08 et Q17 restent des échecs. Référence B0 : 9/48 dans les 10 premières. Seuil inchangé : 18.
2. **B0 contrôlé en entier avant toute variante** (phase 1, `b0_controle.json`), avec le rang historique ; le calcul s'arrête
   au premier écart. Le script vérifie aussi l'égalité des ensembles d'identifiants.
3. **Rang du verdict, le même pour toutes les variantes (B0 compris)** : égalités comptées contre la cible (1 + nombre de
   découpes de score supérieur ou égal, elle exceptée) ; score invalide de la cible (pose sous six articulations) = dernier rang,
   un échec. Dans les fusions, toute découpe au score invalide reçoit le même rang, le dernier : une modalité absente partout
   laisse l'ordre de l'autre inchangé.
4. **K, par sens** : moyenne des cosinus des parties communes si au moins deux, sinon le cosinus B0 de ce sens ; puis le
   meilleur des deux sens. Boîtes de partie : enveloppe des articulations × 1,25, avec une largeur et une hauteur d'au moins un
   quart de la hauteur de la figure ; elles peuvent déborder de la boîte de la figure.
5. **Test du signe bilatéral** (conservateur pour un gain). Publiés aussi : passages dans / hors des 10 premières, dénominateurs,
   nombre de parties communes avec la cible, nombre d'estampes distinctes derrière les découpes oracle (les requêtes ne sont pas
   indépendantes). `composizione` est compté comme `composition`.
6. **Planche écrite avant les chiffres**, tirée parmi les requêtes qui ont une découpe oracle ; je la regarde avant d'ouvrir
   `resume.json`. Galerie sauvegardée au fil du calcul, reprise contrôlée par l'empreinte du script.
