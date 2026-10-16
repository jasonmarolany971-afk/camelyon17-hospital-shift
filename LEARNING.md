# Carnet d'apprentissage

Pour chaque phase : les concepts clés expliqués simplement, puis 3 questions d'entretien avec leurs réponses
(question en français, puis sa formulation en anglais pour s'entraîner aux entretiens internationaux).

---

## Phase 0 — Environnement, données, protocole

### Concepts clés

**1. Le changement de domaine (*domain shift*).**
Un modèle apprend les régularités de ses données d'entraînement, y compris celles qui n'ont rien à voir avec la tâche.
En histopathologie, chaque hôpital a sa recette de coloration (hématoxyline et éosine), son scanner et ses réglages :
les mêmes tissus n'ont pas la même couleur d'un hôpital à l'autre. Un modèle peut utiliser la couleur comme raccourci
et s'effondrer sur un nouvel hôpital. C'est exactement ce que Camelyon17-WILDS mesure.

**2. Pourquoi découper par hôpital.**
Un découpage aléatoire mettrait des patches du même hôpital (et de la même lame) dans le train et le test :
on mesurerait la performance « à la maison », pas sur un nouvel hôpital. WILDS isole donc des hôpitaux entiers :
`train` et `id_val` viennent des hôpitaux 0, 3 et 4 ; `val` de l'hôpital 1 ; `test` de l'hôpital 2.
- `id_val` (*in-distribution*) répond à : « le modèle a-t-il appris ? »
- `val` (*out-of-distribution*) répond à : « le modèle généralise-t-il ? » et sert à **choisir** (époque, seuils).
- `test` répond à : « quelle performance annoncer ? » et ne sert **qu'une fois, à la fin**.
L'écart entre `id_val` et `test` mesure le coût du changement d'hôpital.

**3. Pourquoi ne jamais choisir sur le test.**
Chaque choix fait en regardant le test (meilleure époque, meilleur seuil…) adapte un peu le modèle à cet hôpital précis.
Le chiffre obtenu devient optimiste : il ne prédit plus ce qui se passera sur un vrai nouvel hôpital.
C'est une fuite d'information, même si on n'entraîne jamais sur le test.

**4. Reproductibilité : graine, environnement, matériel.**
La graine (*seed*) fixe le hasard : initialisation des poids, ordre des batchs, augmentations.
Mais à graine égale, les résultats peuvent différer légèrement entre un Mac (MPS) et un GPU NVIDIA (CUDA) :
les calculs en virgule flottante n'y sont pas faits dans le même ordre. D'où trois précautions :
graine passée en paramètre, versions exactes des paquets enregistrées, et plusieurs graines pour mesurer la variabilité.

**5. MPS et mémoire unifiée.**
MPS (*Metal Performance Shaders*) est le moteur qui permet à PyTorch d'utiliser le GPU des puces Apple.
Le CPU et le GPU partagent la même mémoire (16 Go ici) : pas de copie lente entre les deux, mais un plafond commun.
Le M4 du MacBook Air n'a pas de ventilateur : sur un calcul long, il ralentit pour ne pas surchauffer.
D'où la répartition : Mac pour développer, GPU cloud pour les gros entraînements.

**6. Ce qui va dans Git et ce qui n'y va pas.**
Git versionne le code, les configurations et les petits résultats (CSV, figures). Les données (15 Go), les poids
des modèles et les caractéristiques extraites sont régénérables par le code : ils restent hors du dépôt (`.gitignore`).

### Questions d'entretien

**Q1. Qu'est-ce que le *domain shift* en pathologie numérique, et comment Camelyon17-WILDS permet-il de le mesurer ?**
*(EN: What is domain shift in computational pathology, and how does Camelyon17-WILDS measure it?)*

> Les images d'un même type de tissu varient selon l'hôpital : protocole de coloration, scanner, épaisseur des coupes.
> Un modèle entraîné sur quelques hôpitaux peut s'appuyer sur ces différences au lieu de la morphologie des cellules.
> Camelyon17-WILDS sépare les hôpitaux : on entraîne sur trois, on choisit sur un quatrième et on teste sur un cinquième
> jamais vu. L'écart entre la performance sur les hôpitaux d'entraînement et sur l'hôpital de test chiffre l'effet du changement.

**Q2. Pourquoi deux ensembles de validation, et lequel utilisez-vous pour choisir le modèle ?**
*(EN: Why two validation sets, and which one do you use for model selection?)*

> `id_val` vient des mêmes hôpitaux que l'entraînement : il dit si le modèle apprend. `val` vient d'un hôpital non vu :
> il approxime ce qui se passera sur un nouvel hôpital. Je sélectionne sur `val`, comme le protocole officiel WILDS.
> C'est une hypothèse à assumer : on suppose qu'on dispose d'un hôpital externe pendant le développement.
> Le test, lui, ne sert qu'à la mesure finale.

**Q3. Vos résultats varient selon la graine. Comment les présentez-vous ?**
*(EN: Your results vary across random seeds. How do you report them?)*

> Au moins 3 graines par configuration, avec moyenne ± écart-type et les valeurs de chaque graine.
> Les auteurs de WILDS signalent une forte variabilité entre graines sur ce jeu de données : un seul run peut être
> chanceux ou malchanceux. Je ne conclus pas qu'une méthode est meilleure si l'écart est plus petit que la dispersion.
> Trois graines restent peu ; je le mentionne comme limite.

---

## Phase 1 — Données et exploration

### Concepts clés

**1. Provenance des données.**
Le serveur officiel étant hors service, on a utilisé une copie publique. Une copie peut être incomplète ou modifiée,
donc trois réflexes : **épingler la version** (hash de commit : la copie ne peut pas changer entre deux exécutions),
**vérifier contre une source indépendante** (les effectifs publiés dans l'article WILDS, 10 lames par hôpital, aucune
lame partagée entre train et val/test), **tracer** (`results/data_provenance.json`). Limite honnête : sans somme de
contrôle officielle, on ne peut pas prouver que chaque pixel est identique à l'original.

**2. Écrire le marqueur en dernier.**
`wilds` considère le jeu de données présent si `RELEASE_v1.0.txt` existe. On l'écrit à la toute fin : une conversion
interrompue ne peut pas être prise pour une conversion complète. Même idée que les transactions en base de données.

**3. Vérifier par un test, pas par la documentation.**
En phase 0, j'avais conclu à partir de la documentation que `torch-scatter` était inutile. Le test de l'évaluateur
a montré le contraire. La correction : un module de remplacement minimal (`src/wilds_compat.py`) et un test qui
compare son résultat à un calcul indépendant. Le README d'un paquet décrit l'intention ; seul le code exécuté fait foi.

**4. L'unité d'indépendance est la lame, pas le patch.**
Deux patches voisins d'une même lame se ressemblent presque comme deux photos du même objet. Si une lame est des
deux côtés d'un découpage, on mesure en partie de la mémorisation. Ici, `id_val` partage ses 30 lames avec le train :
il est optimiste. Pour toute validation croisée, il faut des **folds groupés par lame** (`GroupKFold`).
Démonstration chiffrée : le « détecteur d'hôpital » fait 76,5 % ± 1,3 % avec des folds par patch, et 71,3 % ± 13,7 %
avec des folds par lame. La fuite gonfle le score **et** fait croire à une stabilité qui n'existe pas.

**5. Le nombre réel d'exemples indépendants.**
455 954 patches, mais dans les hôpitaux H3 et H4, 2 lames fournissent plus de 90 % des patches tumoraux.
Le modèle apprend « la tumeur » à partir d'une poignée de lames : c'est peu, et c'est une explication plausible
de la forte variabilité entre graines signalée par WILDS.

**6. Densité optique et coloration H&E.**
Un colorant absorbe la lumière. La loi de Beer-Lambert dit que la **densité optique** `OD = -log(I / I0)` est
proportionnelle à la quantité de colorant. On peut donc décomposer chaque pixel en contributions d'hématoxyline
(noyaux, violet-bleu) et d'éosine (cytoplasme, rose) : c'est la **déconvolution des couleurs** (Ruifrok & Johnston,
`skimage.color.rgb2hed`). Le fond blanc a une densité optique proche de 0 : un seuil (0,15, critère de Macenko)
isole le tissu.

**7. Séparer l'effet étudié des effets parasites.**
Les patches tumoraux sont plus violets (noyaux denses). Pour comparer la coloration des hôpitaux, on échantillonne
autant de patches tumoraux que non tumoraux par hôpital : sinon, un hôpital riche en tumeurs paraîtrait « plus violet »
sans aucune différence de coloration.

**8. Regarder le test sans tricher.**
Explorer les images de l'hôpital de test est acceptable **parce que** les choix expérimentaux étaient fixés et commités
avant (phase 0). C'est l'idée du **pré-enregistrement** : l'historique Git prouve que les méthodes n'ont pas été choisies
après avoir vu le test.

### Questions d'entretien

**Q1. Le serveur officiel était en panne et vous avez utilisé une copie tierce. Comment savez-vous que vos données sont correctes ?**
*(EN: The official server was down and you used a third-party copy. How do you know your data is correct?)*

> J'ai épinglé la copie à un commit précis, puis vérifié le résultat contre les chiffres publiés dans l'article :
> effectifs exacts de chaque split, 10 lames par hôpital, aucune lame de validation ou de test dans le train. J'ai aussi
> vérifié la structure : fichiers tous présents, chemins uniques, images 96×96. Les PNG sont recopiés octet pour octet,
> sans ré-encodage, et la provenance est enregistrée dans le dépôt. Limite : sans somme de contrôle officielle, je ne
> peux pas prouver l'identité au pixel près ; si le serveur officiel revient, je comparerai les métadonnées.

**Q2. Pourquoi grouper la validation croisée par lame ? Qu'observe-t-on sinon ?**
*(EN: Why group cross-validation by slide? What happens if you don't?)*

> Les patches d'une même lame sont très corrélés. Si une lame est à la fois dans l'entraînement et l'évaluation, le
> modèle est en partie évalué sur ce qu'il a déjà vu. Dans mon exploration, un classifieur qui prédit l'hôpital à partir
> de la couleur obtient 76,5 % ± 1,3 % avec des folds par patch et 71,3 % ± 13,7 % avec des folds par lame : la fuite
> gonfle le score de 5 points et divise l'écart-type par dix, ce qui donne une fausse impression de stabilité.

**Q3. Comment montrez-vous que la différence de couleur entre hôpitaux est réelle, et pas un effet du contenu des images ?**
*(EN: How do you show the color difference between hospitals is real, not an effect of image content?)*

> Trois précautions : un échantillon équilibré tumeur / non-tumeur par hôpital, des statistiques calculées sur les seuls
> pixels de tissu (seuil de densité optique), et une décomposition en hématoxyline / éosine. Résultat : la moitié centrale
> des patches de l'hôpital de test a une concentration d'éosine entièrement au-dessus de celle de chaque hôpital
> d'entraînement, et 5 nombres de couleur suffisent à reconnaître l'hôpital dans 71 % des cas contre 20 % au hasard.
> Limite : la matrice de colorants de `rgb2hed` est générique ; les valeurs sont comparables entre hôpitaux, pas absolues.
