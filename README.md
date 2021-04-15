# Tracking de microbilles et étude de la compression d'une goutte dans un circuit microfluidique

Ce code a permis d'étudier la compression d'une goutte de gel bloquée dans un entonnoir lorsque celle-ci était soumise à des débits variés (cf. images dans le dossier compression150a). Des microbilles introduites dans la goutte ont permis de déterminer la matrice jacobienne de la transformation (appelée abusivement "matrice des déformations" ou "matrice de comression" dans le code) en chaque point de la goutte.

Le procédé utilisé est le suivant :

* On détecte un nombre *nbilles* de billes sur chaque image en utilisant un filtre de convoution adapté à la forme des microbilles que l'on souhaite détecter (ici généralement un carré creux de 3x3 pixels)
* On fait le lien entre les billes détectées indépendamment sur chaque image en utilisant le transport optimal (algorithme hongrois).
* On en déduit la "matrice des déformations" autour de chaque bille en effectuant une régression linéaire sur les déplacements relatifs de ses plus proches voisins par rapport à elle.
* On calcule une triangulation de Delaunay et on effectue une interpolation affine sur chaque triangle pour estimer la valeur de cette matrice en chaque point de la goutte.

Le code est structuré de la façon suivante :

* La classe Débit est sert à gérer les paramètres de chaque film réalisé (correspondant à des débits différents) : nombre d'images, numérotation des images, nombre de billes à détecter... Seul l'un de ces films a pu être uploadé sur GitHub à cause de la limitation en espace mémoire mais nous en avons en réalité étudié 6, d'où l'utilité de cette classe.
* La classe Image sert à gérer les images indépendamment les unes des autres : chargement de l'image, repérage de la forme de la goutte, des microbilles, affichage etc.
* La classe Movie permet de faire le lien entre les différentes images d'un film : tracking des microbilles à l'aide du transport optimal, calcul de la matrice de compression, affichage des trajectoires et de la compression, ...
