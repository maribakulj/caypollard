"""What each step does, in words a reader can check against the code it names."""

from __future__ import annotations

STEPS: dict[str, dict] = {
    "source": {
        "title": "L'image de départ",
        "what": "L'image telle qu'elle est reçue, convertie en RVB. Toutes les étapes partent d'elle, "
        "jamais d'une étape précédente, sauf la silhouette et le pixel-sur-rendu qui partent d'un rendu.",
        "code": "scripts/embed_images.py, scripts/segment_shapes.py",
    },
    "renders": {
        "title": "Rendus indifférents au support",
        "what": "Cinq transformations qui tentent d'effacer ce qui distingue une photographie de tableau "
        "d'une gravure numérisée : le gris (rien de plus que la couleur retirée), les contours "
        "(Sobel après lissage gaussien, normalisé au 99e centile), la forme (réduction à 48×48 puis "
        "étirement des centiles 2–98), la silhouette (la forme seuillée à sa médiane), la masse "
        "(Otsu, puis fraction d'encre locale seuillée).",
        "why": "Mesuré dans docs/VISUAL_RECORD.md « Why pixels will not do » : chaque rendu abaisse la "
        "reconnaissance du corpus par l'encodeur, monotonement ; la silhouette la cache le mieux "
        "(89,3 % contre 97,4 %), Sobel retrouve le mieux le partenaire (rang médian 600 contre 1 303).",
        "code": "scripts/render_medium_invariant.py : render(mode, side=448, sigma=1.2, coarse=48)",
        "params": {
            "render_side": "côté du rendu (448)",
            "render_sigma": "lissage avant Sobel (1,2)",
            "render_coarse": "grille grossière de la forme et de la masse (48)",
        },
    },
    "traits": {
        "title": "Traits : le dessin au trait, puis le croquis, puis le pictogramme",
        "what": "Le contraire des rendus précédents, qui partent du ton. Ici on part des bords : l'image en "
        "gris est lissée (σ = 2 px sur un côté de 320, ce qui efface hachures et coups de pinceau), "
        "on prend la force et la direction du gradient, on ne garde que les crêtes (suppression des "
        "non-maxima) et on les relie par hystérésis (un bord faible ne compte que s'il touche un bord "
        "fort). Résultat : des contours d'un pixel, à deux couleurs, sans aplat. Ils sont amincis, "
        "puis parcourus en traits ; à un croisement, le trait continue par la branche la plus droite "
        "(virage < 60°). Chaque trait est simplifié en polyligne (Douglas-Peucker, 1,5 px). Le "
        "croquis garde les 40 traits les plus longs ; le pictogramme les 12, simplifiés plus fort.",
        "why": "C'est la représentation que la ligne B n'a pas essayée : « abstraire chaque élément à ses "
        "traits », comme un aleph garde du bœuf les quelques lignes qui disent encore bœuf. Le budget "
        "de traits fixe (40, 12) rend l'abstraction comparable d'une image à l'autre, là où un seuil de "
        "longueur effaçait tout sur les images calmes. Un bord n'est pas un ton : une figure claire sur "
        "fond sombre et une sombre sur papier donnent le même contour. Une ligne gravée donne deux bords, "
        "un de chaque côté ; à cette échelle ils fusionnent le plus souvent.",
        "code": "src/caypollard/sketch.py : contours(), thin(), trace(), simplify(), stroke_descriptor() ; "
        "scripts/build_sketch_channel.py (tables du pool : dinov2-sketch, stroke-signs)",
        "params": {
            "sk_sigma": "lissage avant le gradient (2)",
            "sk_high": "centile de force gardé comme bord fort (90)",
            "sk_low": "fraction du seuil fort pour un bord faible (0,4)",
            "sk_turn": "virage maximal à un croisement, degrés (60)",
            "sk_epsilon": "simplification des traits, px (1,5)",
            "sk_sketch": "traits du croquis (40)",
            "sk_pictogram": "traits du pictogramme (12)",
            "sk_pict_epsilon": "simplification du pictogramme, px (4)",
        },
    },
    "pixels": {
        "title": "Pixels : DINOv2",
        "what": "L'image est réduite (petit côté 256, bicubique), recadrée au centre en 224×224, normalisée "
        "(moyennes ImageNet), puis passée dans facebook/dinov2-base ; on garde le jeton CLS "
        "(768 valeurs) et on le normalise en L2. Aucun apprentissage sur ce corpus.",
        "why": "C'est la référence que tout le reste tente de battre. Un plongement photographique nomme "
        "le corpus d'origine 97,4 % du temps : il lit l'objet, l'institution et la main, faiblement "
        "ce qui est représenté (régimes, docs/VISUAL_RECORD.md).",
        "code": "src/caypollard/vision/encoders.py : HuggingFaceVisionEncoder, pooling CLS",
    },
    "silhouette": {
        "title": "Silhouette : DINOv2 sur le rendu",
        "what": "Le rendu « silhouette » est enregistré en JPEG (qualité 92, comme dans l'expérience) puis "
        "encodé par le même DINOv2. La compression fait partie de la chaîne : la table gelée a été "
        "calculée sur les fichiers relus.",
        "code": "scripts/render_medium_invariant.py puis scripts/embed_images.py",
    },
    "segmentation": {
        "title": "Découpe en régions",
        "what": "Gris 256×256, lissage gaussien (σ = 256/64 = 4 px), étirement des centiles 2–98. Le ton "
        "de page est la médiane. Pour chaque polarité (sombre, clair), on garde ce qui s'écarte de "
        "la page de plus de 0,12, on ouvre (3×3), on étiquette les composantes connexes. Une région "
        "vaut si elle couvre au moins 0,4 % de l'image ; on garde les 24 plus grandes.",
        "why": "Classique et non appris, exprès : un segmenteur appris sur des photographies en hérite les "
        "statistiques, c'est-à-dire le support qu'on cherche à fuir. Le document parle d'un "
        "watershed ; le code fait un seuillage à deux polarités, ce qui est noté ici.",
        "code": "scripts/segment_shapes.py : segment(side=256, min_area=0.004, max_regions=24, smoothing=64, threshold=0.12)",
        "params": {
            "seg_side": "côté de travail (256)",
            "seg_min_area": "aire minimale d'une région (0,004)",
            "seg_max_regions": "régions gardées (24)",
            "seg_smoothing": "côté / σ du lissage (64)",
            "seg_threshold": "écart au ton de page (0,12)",
        },
    },
    "descriptors": {
        "title": "Description de chaque région",
        "what": "Seules des propriétés qui survivent à un changement de support : aire, élongation de la "
        "boîte, remplissage de la boîte, ton (plus sombre ou plus clair que la page), trous, "
        "échelle par rapport à la région médiane, quatre invariants de Hu (en log signé), et une "
        "signature radiale sur 24 secteurs (rayon maximal par secteur, divisé par la moyenne, "
        "tournée pour commencer au plus grand rayon). 34 nombres. Ni couleur, ni texture, ni résolution.",
        "code": "scripts/segment_shapes.py (descripteurs) et scripts/build_shape_vocabulary.py : descriptor()",
    },
    "formes": {
        "title": "Formes muettes : le lexique de 256 signes",
        "what": "Chaque descripteur est standardisé (centre et écart-type du corpus d'ajustement) puis "
        "rattaché au plus proche de 256 centres k-means. Le vecteur de l'image est l'histogramme "
        "des signes, pondéré par l'aire de chaque région, normalisé en L2. Les centres ont été "
        "réajustés à l'identique par scripts/fit_demo_models.py et vérifiés contre la table gelée.",
        "why": "« Un signe est une tendance, pas un nom » : l'ascendance Iconclass des régions d'un signe "
        "est significativement plus partagée que le hasard (d = 0,61) mais un treizième d'ascendance "
        "seulement. C'est le canal de l'objet : fort sur la division 2, mauvais sur les scènes.",
        "code": "scripts/build_shape_vocabulary.py ; modèle data/derived/demo/atoms-v2-256.npz",
    },
    "relations": {
        "title": "Relations spatiales : la grammaire",
        "what": "Un vocabulaire plus grossier de 64 signes (réajusté sur toutes les régions du pool), les "
        "dix plus grandes régions, et pour chaque paire ordonnée une relation parmi sept : contient, "
        "dans (inclusion des boîtes), touche (boîtes qui se chevauchent), au-dessus, en-dessous, droite, "
        "gauche (selon le plus grand écart des centroïdes). Un attribut est le triplet (signe, relation, "
        "signe) ; seuls les 9 518 triplets vus au moins 30 fois dans le pool sont gardés, et le "
        "vecteur compte leurs occurrences. Un lion sous une couronne n'est pas un lion qui la porte.",
        "why": "Refusée trois fois par les mesures : sur le corpus large, ajouter les relations aux formes "
        "ne bouge le rang médian que d'une unité sur vingt mille (docs/VISUAL_RECORD.md, « The "
        "grammar, twice asked and twice refused »). Et le vocabulaire des relations est le carré du "
        "lexique : à 256 signes, 14 651 images sur 21 128 n'ont aucun triplet assez fréquent, d'où "
        "les 64 signes. Elle est exposée ici pour être vue avant d'être écartée.",
        "code": "scripts/build_shape_relations.py : relation() ; modèles data/derived/demo/atoms-64.npz, relations-64.npz",
        "params": {"rel_max_regions": "régions considérées (10)"},
    },
    "composition": {
        "title": "Composition : où est la masse",
        "what": "Gris 64×64, écart absolu à la médiane (une figure claire sur fond sombre compte comme une "
        "sombre sur papier), moyenne par blocs 8×8, normalisée à somme 1. Puis : les 64 cases, "
        "8 sommes de lignes, 8 de colonnes, la symétrie gauche-droite et haut-bas, la masse du "
        "centre, l'entropie, et sur le nuage des centroïdes de régions : dispersion, élongation "
        "(log du rapport des valeurs propres), nombre de régions (log). 87 nombres.",
        "why": "C'est le canal de la scène : une Annonciation et une Nativité partagent des figures et de "
        "l'architecture dont les formes locales sont génériques ; ce qu'elles partagent, c'est où "
        "sont les choses. Il échoue sur les vases, dont la forme dicte la mise en page.",
        "code": "scripts/build_composition_channel.py : composition(grid=8)",
        "params": {
            "comp_grid": "taille de la grille (8) — la changer change la dimension, donc rompt la comparaison au pool"
        },
    },
    "repetition": {
        "title": "Répétition : la multiplicité comme quantité typée",
        "what": "Avec un second vocabulaire d'atomes (ajusté sur toutes les régions du pool), on compte "
        "les occurrences de chaque signe. Un profil de 4 cases (signes vus une fois, deux fois, "
        "quelques fois, beaucoup), puis pour chaque signe un indicateur de sa case : 256 × 4. "
        "1 028 nombres, normalisés en L2.",
        "why": "Un Isotype dit « trois » en dessinant trois fois le signe ; un histogramme en fait une "
        "grandeur. Le plus fort canal seul sur les objets, et il nuit aux scènes religieuses à tout "
        "poids : un épisode ne fixe pas son compte.",
        "code": "scripts/build_repetition_channel.py ; modèle data/derived/demo/atoms-pool.npz",
    },
    "groups": {
        "title": "Signes composites : des amas de régions qui se touchent",
        "what": "Deux régions sont « parties d'une même chose » si leur écart de boîtes est petit par "
        "rapport à leur diagonale (marge 0,15) et si leurs aires sont dans un rapport ≤ 3. "
        "Fermeture transitive (union-find). Chaque amas d'au moins 2 parties est décrit par "
        "l'histogramme de ses atomes (pondéré par l'aire, L2) et cinq mesures de contour, "
        "remises à l'échelle, puis rattaché au plus proche de 128 centres. Le vecteur de "
        "l'image compte les amas par composite.",
        "why": "Regrouper a fait tomber la reconnaissance du support de 75,3 % à 61,4 % sans changer la "
        "nommabilité : « la géométrie les joint ; le sens non ». Ce canal est la brique « formes » "
        "du record, ce que la doc ne dit pas et que la vérification numérique a montré.",
        "code": "scripts/build_shape_groups.py (margin=0.15, ratio=3.0) ; modèle data/derived/demo/composites-v3.npz",
        "params": {
            "grp_margin": "écart maximal relatif (0,15)",
            "grp_ratio": "rapport d'aires maximal (3)",
        },
    },
    "record": {
        "title": "Le record : concaténation pondérée",
        "what": "Composites (poids 1), composition (poids 2), répétition (poids ½), chacun normalisé en "
        "L2, concaténés, puis normalisés en L2 ensemble. 128 + 87 + 1 028 = 1 243 nombres. Une "
        "image sans amas n'a pas de record.",
        "why": "Les régimes ont montré que « les canaux sont un ensemble où choisir, pas une somme à "
        "calculer » : ce mélange-là est le meilleur trouvé pour la division 2 (objets), et il est "
        "battu par un canal seul ailleurs.",
        "code": "scripts/build_visual_record.py",
    },
    "palette": {
        "title": "Palette : onze termes de couleur",
        "what": "RVB 128×128, équilibrage grey-world (chaque canal ramené à la moyenne commune), conversion "
        "Lab (D65), chaque pixel rattaché au plus proche de onze termes (noir, blanc, rouge, vert, "
        "jaune, bleu, brun, orange, rose, violet, gris) par distance ΔE76. Fractions, L2.",
        "why": "Ce canal identifie la session de numérisation (61,3 % du volume contre 31 % de plancher). "
        "Poids zéro pour chercher un sujet ; mais la meilleure représentation mesurée pour "
        "reconnaître un type d'objet rare (×5,29 le hasard). Une question, un canal.",
        "code": "scripts/build_surface_channels.py : palette_of(side=128)",
    },
    "signal": {
        "title": "Signal : le grain de la numérisation",
        "what": "Gris 128×128 : histogramme de luminance en 12 cases, moyenne, écart-type, rugosité "
        "(laplacien moyen absolu) et étendue (p95 − p5). 16 nombres, L2.",
        "why": "Construit pour nommer le facteur de confusion directement : grain, accentuation, "
        "compression. Deuxième sur les objets rares, dernier sur le sujet.",
        "code": "scripts/build_surface_channels.py : signal_of(side=128)",
    },
    "nommes": {
        "title": "Nœuds nommés : un nommeur à vocabulaire fermé",
        "what": "On demande à un modèle de vision « les choses distinctes représentées et où chacune se "
        "trouve », avec un vocabulaire fermé de 116 noms ordinaires et neuf places. Ici le "
        "nommeur est Claude (Sonnet) par Claude Code en mode non interactif ; la table gelée a "
        "été nommée par un modèle Mistral, avec la même question. Les trois premiers nœuds sont "
        "comptés (le même mot deux fois compte 2), 116 nombres, L2.",
        "why": "C'est le canal sémantique. Il gagne sur « même sujet, autre objet, autre collection, "
        "autre siècle » (×1,55) et à dire ce qu'un objet de musée est catalogué (hits@1 0,342). "
        "Ses trous : le vocabulaire écrit à la main (moulin, taureau, mouton absents), et un "
        "rappel de 41 % au mieux sur un chat dans une scène de genre.",
        "code": "scripts/name_nodes_vlm.py (prompt, cases, schéma) ; scripts/build_node_record.py (vecteur) ; "
        "src/caypollard/demo/namer.py (Claude Code)",
        "params": {"nodes_max": "nœuds demandés (6)", "nodes_kept": "nœuds comptés (3)"},
    },
    "pose": {
        "title": "Pose : les angles des membres, d'après Impett",
        "what": "Un détecteur de points clés entraîné sur des photographies (Keypoint R-CNN, torchvision) "
        "cherche des personnes ; on garde celles dont le score dépasse 0,75, six au plus, et les "
        "points dont le score dépasse 3. Il faut les deux épaules et une hanche. L'axe du torse "
        "(milieu des épaules vers la hanche) sert de référence : pour neuf segments (bras, "
        "avant-bras, cuisses, jambes, tête) on note le cosinus et le sinus de l'angle relatif et "
        "un indicateur de présence, plus la largeur d'épaules rapportée au torse. Une image = "
        "moyenne et écart-type sur ses figures, et le log de leur nombre : 57 nombres.",
        "why": "Elle gagne partout où il y a des figures humaines (division 3 au rang 65 contre 114 pour le "
        "record) et perd sur les scènes religieuses. Sa vraie limite est la couverture : le "
        "détecteur trouve une figure exploitable dans 28 % des images, gravures comme tableaux. "
        "La table de recherche ne couvre que 1 903 images (769 emblèmes, 1 134 œuvres) : la "
        "chaîne n'a jamais été passée sur tout le pool, ce qui prendrait quelques heures ici.",
        "code": "scripts/build_pose_channel.py : pose_features() ; table data/derived/shapes/pose-partial.npz",
        "params": {
            "pose_score": "score minimal d'une personne (0,75)",
            "pose_keypoint": "score minimal d'un point (3)",
            "pose_max_figures": "figures gardées (6)",
            "pose_max_side": "grand côté avant détection (640)",
        },
    },
    "mix": {
        "title": "Tout mélangé",
        "what": "Nœuds nommés, pixels, palette, signal et formes (vocabulaire v4, équilibré par corpus), "
        "chacun normalisé, concaténés à poids égaux, normalisés. 1 167 nombres.",
        "why": "« Mélanger n'aide que là où rien ne sait la réponse » : sur la matière, seul régime où "
        "aucun canal n'est fort. Partout ailleurs, choisir bat mélanger.",
        "code": "scripts/build_visual_record.py ; modèle data/derived/demo/atoms-v4-256.npz",
    },
    "search": {
        "title": "La recherche",
        "what": "Chaque vecteur est normalisé en L2 ; le score est le produit scalaire, c'est-à-dire le "
        "cosinus. Le pool entier est parcouru (aucun index approché) et les k meilleurs gardés. "
        "Pour mélanger des canaux : chaque vecteur est normalisé, multiplié par son poids, "
        "concaténé, et le tout renormalisé ; le pool est construit de même sur les images "
        "présentes dans tous les canaux choisis. Poids zéro = canal retiré.",
        "why": "Les régimes de docs/VISUAL_RECORD.md sont exactement cette recherche, à laquelle on ajoute "
        "un filtre : ne classer que les candidats que le régime admet (autre type, autre "
        "collection…). Les badges de chaque résultat disent ce qu'il partage avec la requête, et "
        "les notations « moyeux » (portées par plus de 3,2 % du pool) ne comptent pas comme sujet.",
        "code": "src/caypollard/demo/pool.py",
    },
}

DEFAULTS: dict[str, float | int] = {
    "seg_side": 256,
    "seg_min_area": 0.004,
    "seg_max_regions": 24,
    "seg_smoothing": 64.0,
    "seg_threshold": 0.12,
    "render_side": 448,
    "render_sigma": 1.2,
    "render_coarse": 48,
    "comp_grid": 8,
    "surface_side": 128,
    "grp_margin": 0.15,
    "grp_ratio": 3.0,
    "grp_min_parts": 2,
    "nodes_max": 6,
    "nodes_kept": 3,
    "rel_max_regions": 10,
    "sk_sigma": 2.0,
    "sk_high": 90.0,
    "sk_low": 0.4,
    "sk_turn": 60.0,
    "sk_epsilon": 1.5,
    "sk_sketch": 40,
    "sk_pictogram": 12,
    "sk_pict_epsilon": 4.0,
    "pose_score": 0.75,
    "pose_keypoint": 3.0,
    "pose_max_figures": 6,
    "pose_max_side": 640,
}
