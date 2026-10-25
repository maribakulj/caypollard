# Banc extérieur Iconclass gelé : sources possibles et risque de recouvrement avec ICAI

Date : 25/10/2026. Étude de sources seulement, aucun téléchargement important (disque : 2,3 Go libres, commande `df -h /System/Volumes/Data`). Tout ce qui n'est pas vérifié est marqué **[non vérifié]**.

## 1. Rijksmuseum

**Les notations Iconclass sont bien rattachées aux objets (vérifié).**
- Commande : `curl -A "caypollard/0.1 (research prototype)" "https://data.rijksmuseum.nl/search/collection?objectNumber=RP-P-1878-A-1129"` renvoie l'objet `https://id.rijksmuseum.nl/200208896`.
- Puis `curl -L -H "Accept: application/ld+json" https://id.rijksmuseum.nl/200208896` : l'objet pointe vers un `VisualItem` (`https://id.rijksmuseum.nl/202208896`).
- Ce `VisualItem` contient `represents_instance_of_type` -> `https://id.rijksmuseum.nl/2211448`, avec `equivalent` -> `https://iconclass.org/43C111241`. Le code Iconclass est donc au deuxième saut : objet -> VisualItem -> Type -> équivalent iconclass.org. Un seul exemple a été ouvert.
- Le même `VisualItem` donne `digitally_shown_by` -> un `DigitalObject` (l'image), et `subject_to` -> `https://creativecommons.org/publicdomain/mark/1.0/` (Public Domain).

**Passer à l'échelle.**
- Les dumps (https://data.rijksmuseum.nl/docs/data-dumps, page ouverte, version du 2024-11-07) sont en N-Triples, un fichier par ressource, en archives `.tar.gz`. Comptes annoncés : `image` 580 039, `item` 381 376, `instance` 321 354, `work` 321 354, `classification` 23 834 (« avec références à SKOS/ICONCLASS »), `concept` 14 051. Les tailles des archives ne sont pas données sur la page. Je n'ai rien téléchargé.
- Piste la plus légère (non testée) : parcourir l'API de recherche (`https://data.rijksmuseum.nl/search/collection?type=print&imageAvailable=true`, 387 268 résultats vérifiés le jour même, JSON Linked Art paginé), puis résoudre chaque objet à la demande. Il faut compter 3 à 4 requêtes par objet (objet, VisualItem, Type, image). C'est lent à 380 000 objets : il vaut mieux échantillonner, ou lire les dumps `item`/`instance`/`classification` sur Kaggle ou Colab, pas en local. Dans le dump `classification`, repérer les Types dont l'`equivalent` est une URL `iconclass.org` donne la table Type -> code.
- La documentation de l'API de recherche (https://data.rijksmuseum.nl/docs/search, ouverte) ne liste aucun paramètre `iconclass` ni `subject` (paramètres : aboutActor, creator, creationDate, description, imageAvailable, material, memberOfSetId, objectNumber, technique, title, type). On ne peut donc pas filtrer « a un code Iconclass » côté serveur.
- Autre voie : OAI-PMH et LDES sont cités sur https://data.rijksmuseum.nl/docs/ ; je n'ai pas vérifié si le XML OAI contient les codes **[non vérifié]**.

**Nombre d'objets avec codes Iconclass : inconnu [non vérifié].** Aucune source ouverte ne le donne. Un outil tiers (https://github.com/kintopp/rijksmuseum-iconclass-mcp) dit tenir une base de comptes par notation (« iconclass-counts.db ») mais ne publie pas le total. Ordre de grandeur à mesurer, pas à croire : le dump a 321 354 « work » ; les codes Iconclass portent surtout sur estampes, dessins et tableaux. La première mesure à faire : échantillon de 300 objets tirés au hasard de l'API de recherche, compter ceux dont le Type a un équivalent iconclass.org.

**Licences.**
- Politique officielle (https://data.rijksmuseum.nl/policy, ouverte) : domaine public ou CC0 ; CC BY 4.0 dans d'autres cas ; certaines photos d'œuvres encore protégées ne sont pas libres. La licence est indiquée par objet (`subject_to` dans le JSON). Pour un banc gelé : ne garder que les objets marqués Public Domain / CC0.
- Métadonnées : CC0 (l'objet `subject_of` du VisualItem pointe vers `https://creativecommons.org/publicdomain/zero/1.0/`, vérifié sur l'exemple).
- Attribution demandée mais non obligatoire (même page).

**Résolution des images.** IIIF Image API via Micrio : racine `https://iiif.micr.io`, exemples `https://iiif.micr.io/RFwqO/info.json` et `.../full/max/0/default.png` (https://data.rijksmuseum.nl/docs/iiif/image, ouverte). La résolution maximale réelle n'est pas donnée par la page ; elle se lit dans `info.json` de chaque image **[non vérifié]**. Pour le banc, `/full/!1024,1024/0/default.jpg` suffit et évite le poids. Limites de débit : non documentées, rester à ≥ 1 requête/s **[non vérifié]**.

## 2. Emblematica Online, état local (lecture seule)

Commandes : script Python en lecture seule sur `/Users/marcel/caypollard/data/raw/emblematica` avec les mêmes motifs que `/Users/marcel/caypollard/src/caypollard/datasets/emblematica.py` (balises `iconclass` puis `notation`).
- `xml/` : 31 041 fiches d'emblèmes ; `index.jsonl` : 33 233 lignes ; `books.jsonl` : 1 407 livres.
- Fiches avec au moins une notation Iconclass : **27 216** (280 733 notations au total, doublons inclus). Attention : ce test n'enlève pas les commentaires XML comme le fait `parse_emblem_xml` (commentaire modèle « Please replace with iconclass headings »), donc 27 216 est un maximum. À refaire avec le parseur réel avant de citer **[non vérifié]**.
- Images locales (`images/`) : **13 428** picturae ; **les 13 428 ont une fiche avec notation** (recoupement par identifiant `Exxxxxx`). C'est le nombre déjà utilisable sans rien télécharger. Le reste (environ 13 800 fiches annotées) nécessiterait de télécharger les images depuis `emblemimages.library.illinois.edu` (URL dans `index.jsonl`).
- Licence : non vérifiée pour ces données **[non vérifié]** (à lire sur le site Emblematica, https://emblematica.library.illinois.edu).

## 3. Autres sources

Les recherches web n'ont rien renvoyé de précis sur ces sources ; tout ce qui suit est donc **[non vérifié]** sauf mention.
- Arkyves (agrégateur d'ICAI) liste ses partenaires (https://arkyves.org/help/partners, ouverte) : Rijksmuseum, RKD, Virtuelles Kupferstichkabinett, Emblematica (UIUC + HAB), Glasgow, Bibliothèques Virtuelles Humanistes, KB La Haye, Utrecht Love Emblems, etc. Aucune licence d'image n'y est indiquée. Ce sont les mêmes sources que l'on voudrait pour le banc, donc elles sont toutes exposées au recouvrement (section 4).
- RKD (RKDimages) : annotations Iconclass, mais images souvent sous droits ; l'accès par API et la licence n'ont pas pu être confirmés.
- Virtuelles Kupferstichkabinett : annoté Iconclass d'après Arkyves ; protocole d'accès et licence non confirmés.
- Bildindex, Warburg Iconographic Database : non confirmés.
- NGA : je n'ai pas de raison de croire que ses données utilisent Iconclass ; rien vérifié, probablement à écarter.
- Europeana : une notice Rijksmuseum (https://www.europeana.eu/item/90402/RP_F_2014_7_1_10, trouvée via la recherche, non ouverte par moi) montrerait des URI Iconclass dans le champ sujet ; Europeana est donc un moyen de rassembler plusieurs musées avec un seul API. Non testé.
- ICARUS (Springstein 2024) : je n'ai pas retrouvé cette référence par la recherche web **[non vérifié]** ; la note de la consigne (Rijksmuseum et Emblematica inclus) est reprise telle quelle, sans confirmation de ma part.

## 4. Risque de recouvrement avec ICAI

**Constats sur `/Users/marcel/caypollard/data/raw/iconclass-testset/data.json`** (87 744 entrées, dictionnaire nom de fichier -> codes ; script Python `-I` en lecture) :
- 14 531 noms `IIHIM_RIJKS_<nombre>.jpg` : Rijksmuseum. Le nombre est un hachage (ex. `IIHIM_RIJKS_1155186918.jpg`), pas un numéro d'objet : **aucune jointure par identifiant possible** avec `RP-P-...` ou l'ID Linked Art.
- 48 993 noms `IIHIM_<nombre>.jpg` : source non identifiable par le nom (probablement RKD, Virtuelles Kupferstichkabinett, etc. **[non vérifié]**).
- Famille Emblematica / livres d'emblèmes par préfixe (`embhab_`, `embstma_`, `embmne_`, `embepu_`, `uiuc_`, `folger_ill`, `gri`, `norris_`, `loc_rosenwald`, `uva_`) : 7 072 images. Autres lots : `bsb` 5 968, `ursicula` 4 430, `biblia_sacra` 2 793, `dpd` 419.
- Les noms Emblematica d'ICAI (`embhab_xb4463_pic157`) et nos identifiants (`E002924`) n'ont pas de clé commune : jointure par identifiant impossible aussi.

**Détection pratique (peu coûteuse).**
1. Rijksmuseum : l'ID n'est pas retrouvable, donc comparer les images. Hachage perceptuel (pHash/dHash 64 bits) sur miniatures 64 px, ou plongements DINOv2 ; les 14 531 miniatures ICAI sont locales dans le dossier `iconclass-testset`. Pour le candidat, télécharger une miniature IIIF (`!256,256`) et calculer la distance de Hamming (seuil à calibrer, par exemple ≤ 6 sur 64 bits) ou cosinus ≥ 0,95. À calibrer sur des paires connues.
2. Emblematica : même méthode contre les 7 072 images à préfixe livre d'emblèmes. Les mêmes pages sont aussi présentes en doublons dans plusieurs éditions : un exclure-par-voisin-proche (près-duplicata) évite aussi les copies d'éditions différentes. Utile aussi pour ne pas refaire les couples de copies (cf. `motifs-copies.md`).
3. Codes Iconclass identiques et titre : critère secondaire seulement, pas suffisant.

**Combien survivraient ? Non mesurable sans tirer l'échantillon [non vérifié].** Ordres de grandeur, à confirmer :
- Le Rijksmuseum a 387 268 estampes avec image (API, vérifié) ; ICAI en contient au plus 14 531 de toutes sortes. Si ICAI est un échantillon, la plupart des objets Rijksmuseum annotés seraient absents d'ICAI ; mais Arkyves n'a probablement pris que les objets annotés Iconclass, donc le recouvrement est élevé parmi les objets annotés, et surtout un banc extérieur Rijksmuseum partage le même style d'annotation.
- Emblematica local : 13 428 images, dont une part est vraisemblablement dans ICAI (7 072 images à préfixe livre d'emblèmes, mais toutes ne viennent pas d'Emblematica). Estimation à faire par hachage ; on peut avoir 20 à 60 % de recouvrement, chiffre sans base, à remplacer par la mesure.
- Même sans doublon d'image, le **recouvrement de distribution** reste : mêmes catalogueurs, mêmes conventions de codage. Un banc extérieur est « disjoint » en images, pas indépendant en étiquettes. Le dire dans l'article.

## Recommandation

1. Mesurer d'abord (une heure, peu de poids) : 300 objets Rijksmuseum tirés de l'API, part ayant un code Iconclass, part sous Public Domain/CC0, miniatures IIIF 256 px, hachage contre les 14 531 `IIHIM_RIJKS` d'ICAI. Cela donne le taux de survie réel et le nombre d'objets utilisables.
2. Banc Emblematica prêt à l'emploi : les 13 428 picturae locales, après passage du parseur réel et hachage contre les 7 072 images ICAI de la famille emblèmes.
3. Geler la liste (identifiants + codes + somme de contrôle) avant toute évaluation.
