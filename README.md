# microbeadsTracking
Code to visualize the compression of a droplet in a microfluidic circuit by tracking microbeads inside it.

Ce code a permis d'étudier la compression d'une goutte de gel bloquée dans un entonnoir lorsque celle-ci était soumise à des débits variés (cf. images dans le dossier compression150a). Des microbilles introduites dans la goutte ont permis de déterminer la matrice jacobienne de la transformation (appelée abusivement "matrice des déformations" dans le code) en chaque point de la goutte.

Le procédé utilisé est le suivant :

* On détecte un nombre $nbilles$ de billes sur chaque image en utilisant un filtre de convoution adapté à la forme des microbilles que l'on souhaite détecter (ici généralement un carré creux de 3x3 pixels)
