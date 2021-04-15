import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from math import floor
from scipy.signal import convolve2d
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from scipy.spatial import Delaunay
from copy import deepcopy

xmin, xmax, ymin, ymax = 400, 850, 200, 650 # Fenêtre dans laquelle se situe la goutte
x0, y0 = 600, 400 # Un point proche du centre de la goutte

# Filtre utilisé pour détecter les microbilles
filtre = np.array([[0., 1, 0, 1, 0],
                   [1.,-1,-1,-1, 1],
                   [0.,-1, 0,-1, 0],
                   [1.,-1,-1,-1, 1],
                   [0., 1, 0, 1, 0]])

# Noyau de convolution pour le lissage de la forme de la goutte
kernelSize = 10
noyau = np.zeros((ymax-ymin, xmax-xmin))
dx, dy = (xmax-xmin)//2, (ymax-ymin)//2
noyau[dy-kernelSize:dy+kernelSize+1, dx-kernelSize:dx+kernelSize+1] = np.ones((2*kernelSize+1, 2*kernelSize+1))
noyau = np.fft.ifftshift(noyau)
noyau = np.fft.fft2(noyau)

class Debit:
    """Configuration des paramètres utilisés pour le traitement de chacun des films"""
    _debits = []
    _lastPics = []
    _nbilles = []
    _steps = []
    _droites = []
    
    def __init__(self, d):
        self.d = d
        self.debit = Debit._debits[d]
        self.lastPic = Debit._lastPics[d]
        self.nbilles = Debit._nbilles[d]
        self.step = Debit._steps[d]
        self.droites = Debit._droites[d]
        self.n = self.lastPic // self.step
    
    def kToID(self, k):
        return (k + 1) * self.step
    
    def IDToK(self, imageID):
        return imageID // self.step - 1
    
    def setLastPic(self, lP):
        self.lastPic = lP
        self.n = self.lastPic // self.step

class Image(Debit):
    """Chaque instance de cette classe représente une image de goutte"""
    
    def quatreChiffres(n):
        """Retourne une chaine de caractères de taille 4 représentant l'entier n (supposé < 10⁴)"""
        return (4 - len(str(n))) * '0' + str(n)
    
    def fname(self, imageID):
        """Retourne le chemin relatif pour accéder à l'image numéro ID du film en cours de traitement"""
        movie = 'compression' + self.debit
        movie = movie + '/' + movie
        return movie + Image.quatreChiffres(imageID) + '.tif'

    def __init__(self, d, imageID):
        super().__init__(d)
        assert imageID <= self.lastPic, "L'image numéro {} n'existe pas pour le débit {}".format(imageID, self.debit)
        self.numpy = io.imread(self.fname(imageID))
        self.imageID = imageID
        # Champs définis dans d'autres fonctions :
        # goutte
        # convol
        # used
        # billes
        
    def traceDroite(self, droite):
        """Trace une droite noire entre les points p1 et p2 sur le tableau numpy"""
        x1, y1, x2, y2 = droite
        assert x1 != x2, "Les abscisses des deux points doivent être distinces"
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        m = (y2 - y1) / (x2 - x1)
        for x in range(x1, x2 + 1):
            y = floor(y1 + m * (x - x1))
            self.numpy[y, x] = 0
            self.numpy[y+1, x] = 0
    
    def initGoutte(self):
        """Détermine la forme de la goutte"""
        
        for droite in self.droites:
            self.traceDroite(droite)
        
        # Parcours en profondeur pour déterminer la forme de la goutte
        
        self.goutte = np.zeros((1024,1024), dtype = bool)
        L = [(x0, y0, 255)] # x, y, min(couleurs) ; min(couleurs) n'est pas utilisé dans cette version
        A = 0

        while len(L) > 0:
            x, y, b = L.pop()
            if not self.goutte[y,x] and self.numpy[y, x] > 140:
                b1 = min(b, self.numpy[y, x])
                #goutte[y,x] = 0
                self.goutte[y,x] = True
                A += 1
                L.append((x+1,y,b1))
                L.append((x-1,y,b1))
                L.append((x,y-1,b1))
                L.append((x,y+1,b1))

        # Convolution pour lisser : combler les trous et "limer" le bord de la goutte
        
        self.goutte = self.goutte.astype(np.float64)
        gouttefft = np.fft.fft2(self.goutte[ymin:ymax, xmin:xmax])
        #self.goutte = convolve2d(self.goutte, np.ones((0,10)), mode='same')
        gouttefft = gouttefft * noyau
        self.goutte[ymin:ymax, xmin:xmax] = np.fft.ifft2(gouttefft).real
        self._goutte = self.goutte[ymin:ymax, xmin:xmax]
        self.goutte = np.where(self.goutte > 310, np.ones((1024,1024), dtype = bool), np.zeros((1024,1024), dtype = bool))
        
    def reperageBilles(self):
        """Détection des billes à l'aide du filtre 'filtre'"""
        
        dmin = 2 # Distance minimale entre 2 billes
        
        self.convol = convolve2d(self.numpy[ymin:ymax, xmin:xmax], filtre, mode='same')
        argsorted = np.argsort(self.convol.flatten())

        self.used = np.zeros((1024,1024), dtype = bool) # Pour ne pas détecter des billes trop proches les unes des autres
        self.billes = []
        i = argsorted.size - 1

        while len(self.billes) < self.nbilles:
            j = argsorted[i]
            x, y = xmin + j % (xmax-xmin), ymin + j // (xmax-xmin)
            if self.goutte[y,x] and not self.used[y,x]:
                self.billes.append([x, y])
                for dx in range(-dmin, dmin+1):
                    for dy in range(-dmin, dmin+1):
                        self.used[y+dy,x+dx] = True
            i -= 1
    
    def initBW(self):
        """Prépare une copie en noir et blanc de l'image que l'on peut 'colorier'"""
        self.numpyBW = deepcopy(self.numpy)
    
    def initRGB(self):
        self.numpyRGB = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for c in range (3):
            self.numpyRGB[:,:,c] = deepcopy(self.numpy)
    
    def show(self, im='VO', x = (xmin + xmax)//2, y = (ymin + ymax) // 2, a = (xmax-xmin)//2, save=False):
        xmin0, xmax0, ymin0, ymax0 = x - a, x + a, y - a, y + a
        plt.rcParams['figure.figsize'] = [50/2.54, 40/2.54]
        if im == 'VO':
            io.imshow(self.numpy[ymin0:ymax0, xmin0:xmax0])
        elif im == 'BW':
            io.imshow(self.numpyBW[ymin0:ymax0, xmin0:xmax0])
        elif im == 'RGB':
            io.imshow(self.numpyRGB[ymin0:ymax0, xmin0:xmax0, :])
        else:
            print("Invalid argument : im")
            raise
        if save:
            plt.savefig('out.jpg')
        plt.show()
    
    def afficheBille(self, x, y, c = [], a = 2, f = 'o'):
        """c = couleur, a = taille de la boîte, f = forme (. = carre plein, o = carre vide, x = croix)"""
        if len(c) == 0:
            c = np.random.randint(0, 255, size = (3,))
        for dx in range(-a, a+1):
            for dy in range(-a, a+1):
                if f == '.' or (f=='o' and (abs(dx) == 2 or abs(dy)==2)) or f=='x' and abs(dx) == abs(dy):
                    self.numpyRGB[y+dy, x+dx,:] = c

d = 2
seuil = 10
seuil2 = 4

def cisaillement(A):
    trOrder = list(range(A.ndim-2)) + [A.ndim-1] + [A.ndim-2]
    B = np.matmul(np.transpose(A, axes=trOrder), A)
    res = np.trace(B, axis1=-2, axis2=-1)/np.linalg.det(A) - 2
    return np.sqrt(res)

def couleur(niveau, a=1, b=0): # FIXME : a supprimer
    c = 255 * a * niveau + b
    return min(255, max(0, c))

class Movie(Debit):
    """Chaque instance de cette classe représente un film associé à un débit particulier."""
    
    def __init__(self, d):
        super().__init__(d)
        self.trajets_v = dict()
        self.billesValides_v = dict()
        self.cnorm = colors.Normalize(vmin=0.5, vmax=1.5, clip=True)
        self.cmap = cm.get_cmap(name='Reds')
        # Champs de cette classe :
        # k
        # trajets, trajets2, trajets_v
        # billesValides, _billesValides, billesValides2, billesValides_v
        # plusProchesVoisins
        # J, detJ
        # cnorm, cmap

def Movie_billes(self, imageID):
    """Retourne un tableau numpy avec les billes trouvées pour l'image imageID"""
    image = Image(self.d, imageID)
    image.initGoutte()
    image.reperageBilles()
    return np.array(image.billes)

Movie.billes = Movie_billes

def Movie_calculTrajectoires(self):
    # Pour suivre la trajectoire de chaque goutte :
    self.trajets = np.zeros((self.nbilles, self.n, 2), dtype = np.int32)
    self.trajets_v['1'] = self.trajets
    # Billes sans anomalies de trajectoire :
    self.billesValides = np.ones(self.nbilles, dtype = bool)
    self.billesValides_v['1'] = self.billesValides
    # Calcul des trajectoires :
    self.trajets[:,0,:] = self.billes(self.step)
    self.k = 1
    while (self.k < self.n):
        self._nextPosition()
        self.k += 1
    self._billesValides = deepcopy(self.billesValides) # Garde une copie de l'état avant suppression des taches

Movie.calculTrajectoires = Movie_calculTrajectoires

def Movie_nextPosition(self):
    """Calcul la position suivante de chacune des billes en utilisant l'algorithme hongrois"""

    print(self.k, end=' ')
    imageID = self.kToID(self.k)
    currentBilles = self.billes(imageID)
    distances = np.zeros((self.nbilles, self.nbilles))

    for i in range(self.nbilles):
        for j in range(self.nbilles):
            if self.billesValides[i]:
                dij = np.linalg.norm(currentBilles[j] - self.trajets[i, self.k - 1,:])
                if dij < seuil:
                    distances[i,j] = dij
                else:
                    distances[i,j] = 100
            else:
                distances[i,j] = 1

    _, assignment = linear_sum_assignment(distances)

    for i in range(self.nbilles):
        self.trajets[i, self.k, :] = currentBilles[assignment[i]]
        if distances[i, assignment[i]] >= seuil:
            self.billesValides[i] = False

Movie._nextPosition = Movie_nextPosition

def Movie_suppressionTaches(self):
    """Suppression des billes qui n'ont pas bougées -> taches"""
    eps = 10
    for i in range(self.nbilles):
        if self.billesValides[i]:
            if self.dist(i, -1, i, 0) < eps:
                self.billesValides[i] = False

Movie.suppressionTaches = Movie_suppressionTaches

def Movie_deplacementMedian(self, k, bille, a=50):
    """Déplacement médian des billes au sein d'un carré de taille 2*a"""
    x0, y0 = self.coordonnees(bille, self.kToID(k))
    DU = []
    for i in range(self.nbilles):
        x, y = self.coordonnees(i, self.kToID(k))
        if self.billesValides[i] and x >= x0 - a and x <= x0 + a and y >= y0 - a and y <= y0 + a:
            DU.append(self.vecteur(i, k+1, i, k))
    return np.median(DU, axis=0)

Movie.deplacementMedian = Movie_deplacementMedian

def Movie_affinerTrajectoires(self):
    """Affine la trajectoire des billes en fonction des résultats du premier passage"""
    self.trajets2 = np.zeros((self.nbilles, self.n, 2), dtype = np.int32)
    self.trajets2[:,0,:] = self.trajets[:,0,:]
    self.trajets_v['2'] = self.trajets2
    self.billesValides2 = deepcopy(self.billesValides)
    self.billesValides_v['2'] = self.billesValides2
    self.k = 1
    while (self.k < self.n):
        self._nextPosition2()
        self.k += 1

Movie.affinerTrajectoires = Movie_affinerTrajectoires

def Movie_nextPosition2(self):
    """Calcul la position suivante de chacune des billes en utilisant l'algorithme hongrois"""

    print(self.k, end=' ')
    imageID = self.kToID(self.k)
    currentBilles = self.billes(imageID)
    distances = np.zeros((self.nbilles, self.nbilles))

    for i in range(self.nbilles):
        if self.billesValides2[i]:
            positionEstimee = self.trajets2[i, self.k - 1,:] + self.deplacementMedian(self.k-1, i)
        for j in range(self.nbilles):
            if self.billesValides2[i]:
                dij = np.linalg.norm(currentBilles[j] - positionEstimee) ** 2
                if dij < seuil2:
                    distances[i,j] = dij
                else:
                    distances[i,j] = 50
            else:
                distances[i,j] = 1

    _, assignment = linear_sum_assignment(distances)

    for i in range(self.nbilles):
        self.trajets2[i, self.k, :] = currentBilles[assignment[i]]
        if distances[i, assignment[i]] >= seuil2:
            self.billesValides2[i] = False  

Movie._nextPosition2 = Movie_nextPosition2

def Movie_getPosition(self, i, k, v='1'):
    return self.trajets_v[v][i, k, :]

Movie.getPosition = Movie_getPosition

def Movie_coordonnees(self, bille, imageID, v='1'):
    k = self.IDToK(imageID)
    return self.getPosition(bille, k, v=v)

Movie.coordonnees = Movie_coordonnees

def Movie_vecteur(self, i1, k1, i2, k2, v='1'):
    """Vecteur AB où A est la position de i2 à l'instant k2 et B est la position de i1 à l'instant k1"""
    return self.trajets_v[v][i1, k1, :] - self.trajets_v[v][i2, k2, :]

Movie.vecteur = Movie_vecteur

def Movie_dist(self, i1, k1, i2, k2, v='1'):
    """Distance entre la position de i2 à l'instant k2 et la position de i1 à l'instant k1"""
    return np.linalg.norm(self.vecteur(i1, k1, i2, k2, v=v))

Movie.dist = Movie_dist

def Movie_isBilleValide(self, bille, v='1'):
    return self.billesValides_v[v][bille]

Movie.isBilleValide = Movie_isBilleValide

def Movie_calculPlusProchesVoisins(self, v='1'):
    """Calcul des plus proches voisins de chaque point"""
    
    bv = np.where(self.billesValides_v[v])[0]
    self.plusProchesVoisins = np.zeros((self.nbilles, len(bv) - 1), dtype = np.int32)

    for i in bv:
        distances = []
        for j in bv:
            if j!=i:
                distances.append((j, self.dist(i,0,j,0)))
        distances = [x[0] for x in sorted(distances, key=lambda x: x[1])]
        self.plusProchesVoisins[i, :] = np.array(distances)

Movie.calculPlusProchesVoisins = Movie_calculPlusProchesVoisins
            
def Movie_calculMatriceDeformation(self, nvoisins = 4, v='1'):
    
    bv = np.where(self.billesValides_v[v])[0]
    assert nvoisins <= len(bv) - 1, "Il n'y a pas suffisamment de billes valides"

    self.J = np.zeros((self.nbilles, 2, 2))

    for i in bv:
        ppv = self.plusProchesVoisins[i, :nvoisins]
        ref = i * np.ones(nvoisins, dtype=np.int32)
        BT = self.vecteur(ppv, 0, ref, 0, v=v)
        CT = self.vecteur(ppv, -1, ref, -1, v=v)
        AT, _, _, _ = np.linalg.lstsq(BT, CT, rcond=None)
        self.J[i,:,:] = AT.T

    self.detJ = np.linalg.det(self.J)

Movie.calculMatriceDeformation = Movie_calculMatriceDeformation

def Movie_showCompression(self, imageID, v='1', indic = 'det'):
    """FIXME : N'utilise pas encore le résultat de l'affinement de trajectoires"""
    k = self.IDToK(imageID)
    bv = np.where(self.billesValides_v[v])[0]
    points = self.trajets_v[v][bv,k,:]
    delaunay = Delaunay(points) # Triangulation
    print("Triangulation terminée")
    image = Image(self.d, imageID)
    image.initRGB()
    for simplex in delaunay.simplices:
        coords = points[simplex]
        T = np.ones((3,3))
        T[:2,:] = coords.T
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        for x in range(xmin, xmax+1):
            for y in range(ymin, ymax+1):
                lmbda = np.linalg.solve(T, np.array([x, y, 1]))
                if np.all(lmbda >= -1e-6) and np.all(lmbda <= 1+1e-6):
                    J = sum([lmbda[i] * self.J[bv,:,:][simplex[i]] for i in range(3)])
                    if indic == 'det':
                        image.numpyRGB[y, x, :] = self.cmap(self.cnorm(np.linalg.det(J)), bytes=True)[:3]
                    elif indic == 'cis':
                        image.numpyRGB[y, x, :] = self.cmap(self.cnorm(cisaillement(J)), bytes=True)[:3]
    image.show(im='RGB')
    self._image = image

Movie.showCompression = Movie_showCompression

def Movie_showTrajectoires(self, enVert = [], v='1', random=False, save=False):
    bv = np.where(self.billesValides_v[v])[0]
    image = Image(self.d, self.step)
    image.initRGB()
    detJnorm = self.cnorm(2 - self.detJ)
    c = self.cmap(detJnorm, bytes=True)
    for point in bv:
        c1 = c[point,:3]
        if random:
            c1 = np.random.randint(0, 255, size=3)
        if point in enVert:
            c1 = [0, 255, 0]
        for imageID in range(self.step, self.lastPic, self.step):
            x, y = self.coordonnees(point, imageID, v=v)
            image.afficheBille(x, y, c=c1, a=1, f='.')
            
    image.show(im='RGB', save=save)
    self._image = image

Movie.showTrajectoires = Movie_showTrajectoires

def Movie_showTrajectoire(self, bille, start = 'min', stop = 'max', s=1, nvoisins='all', static=False, v='1'):
    """Requiert que plusProchesVoisins soit initialisé, sauf si all==None"""
    bV = self.billesValides_v[v]
    # Initialisation et interprétation des arguments
    if start == 'min':
        start = 0
    else:
        start = self.IDToK(start)
    if stop == 'max':
        stop = self.n
    else:
        stop = self.IDToK(stop)
    if nvoisins == 'all':
        nvoisins = np.count_nonzero(np.where(bV)[0]) - 1
    if nvoisins != None:
        voisinnage = [bille] + list(self.plusProchesVoisins[bille,:nvoisins])
    else:
        voisinnage = [] # Si all==None on affiche l'image brute, sans repérage des billes
    x0, y0 = self.coordonnees(bille, self.step, v=v)
    c = np.random.randint(0, 255, size = (self.nbilles, 3))
    # Parcours des images
    for k in range(start, stop, s):
        imageID = self.kToID(k)
        image = Image(self.d, imageID)
        image.initRGB()
        for point in range(self.nbilles):
            if (nvoisins == 'all' or point in voisinnage) and bV[point]:
                x, y = self.coordonnees(point, imageID, v=v)
                image.afficheBille(x, y, c=c[point])
        if not static: # Si static vaut True la fenêtre est immobile, sinon elle suit la bille
            x0, y0 = self.coordonnees(bille, imageID, v=v)
        print(imageID)
        image.show(im='RGB', x=x0, y=y0, a=50)

Movie.showTrajectoire = Movie_showTrajectoire

def Movie_showSumOfDistances(self):
    sumOfDistances = np.zeros(self.n - 1)
    for k in range(1, self.n):
        sumOfDistances[k-1] = np.sum(np.linalg.norm(self.trajets[:, k,:] - self.trajets[:, 0,:], axis = 1), where = self.billesValides)
    plt.plot(range(2 * self.step, self.lastPic + 1, self.step), sumOfDistances)
    plt.show()

Movie.showSumOfDistances = Movie_showSumOfDistances