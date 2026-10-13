"""The workshop: interchangeable bricks run on the same test benches.

caypollard is a place to try new tools on our corpora, not a system that settles on one.
Every tool is a *brick* of one kind -- it finds the figures in a picture, places their
body points, or turns a whole picture into a vector -- and every brick is scored on the
same *benches*, whose right answers are known in advance (the editions of one Holbein
scene; one subject across media). Results land in one table, so a new brick is judged
against the old ones on the same pictures and the same question.

``scripts/atelier.py`` is the command line; ``bricks`` lists what can be plugged in,
``benches`` what it is scored on, ``measure`` turns descriptions into distances.
"""
