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

**4. L'unité d'indépendance : le patient, pas le patch.**
Deux patches voisins d'une même lame se ressemblent presque comme deux photos du même objet. Si une lame est des
deux côtés d'un découpage, on mesure en partie de la mémorisation. Ici, `id_val` partage ses 30 lames avec le train :
il est optimiste. Un niveau plus haut, les 50 lames viennent de 43 patients (6 patients ont 2 ou 3 ganglions) :
l'unité la plus stricte est donc le **patient**. Pour toute validation croisée : **folds groupés par patient**
(`GroupKFold`, `StratifiedGroupKFold`).
Démonstration chiffrée avec le « détecteur d'hôpital » : 76,5 % ± 1,3 % avec des folds par patch, 71,3 % ± 13,7 %
par lame, 70,5 % ± 10,8 % par patient. La fuite gonfle le score **et** fait croire à une stabilité qui n'existe pas.

*Pourquoi l'écart-type grandit-il ?* Il reflète le nombre d'unités **indépendantes** dans chaque fold de test.
Avec des folds par patch, chaque fold contient environ 2 000 patches tirés des 50 lames : tous les folds se ressemblent,
donc les scores aussi. Avec des folds par patient, chaque fold ne contient que 8 ou 9 patients : le score dépend de
*quels* patients y tombent, et les patients diffèrent beaucoup entre eux. Le grand écart-type n'est pas un défaut de
la méthode, c'est l'incertitude réelle quand on ne dispose que de 43 patients. Avec 5 folds seulement, l'écart-type
est lui-même imprécis : 13,7 contre 10,8 ne se compare pas, 1,3 contre ~11 oui.

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

**Q2. Pourquoi grouper la validation croisée par patient ? Qu'observe-t-on sinon ?**
*(EN: Why group cross-validation by patient? What happens if you don't?)*

> Les patches d'une même lame, et les lames d'un même patient, sont corrélés. Si un patient est à la fois dans
> l'entraînement et l'évaluation, le modèle est en partie évalué sur ce qu'il a déjà vu. Dans mon exploration, un
> classifieur qui prédit l'hôpital à partir de la couleur obtient 76,5 % ± 1,3 % avec des folds par patch,
> 71,3 % ± 13,7 % par lame et 70,5 % ± 10,8 % par patient. La fuite gonfle le score d'environ 6 points et rend
> l'écart-type presque dix fois trop petit : une fausse impression de stabilité. Le chiffre à rapporter est celui
> groupé par patient, parce qu'en pratique le modèle sera appliqué à de nouveaux patients.

**Q3. Comment montrez-vous que la différence de couleur entre hôpitaux est réelle, et pas un effet du contenu des images ?**
*(EN: How do you show the color difference between hospitals is real, not an effect of image content?)*

> Trois précautions : un échantillon équilibré tumeur / non-tumeur par hôpital, des statistiques calculées sur les seuls
> pixels de tissu (seuil de densité optique), et une décomposition en hématoxyline / éosine. Résultat : la moitié centrale
> des patches de l'hôpital de test a une concentration d'éosine entièrement au-dessus de celle de chaque hôpital
> d'entraînement, et 5 nombres de couleur suffisent à reconnaître l'hôpital dans 70,5 % des cas contre 20 % au hasard
> (validation croisée groupée par patient).
> Limite : la matrice de colorants de `rgb2hed` est générique ; les valeurs sont comparables entre hôpitaux, pas absolues.

---

## Phase 2 — CNN de référence et évaluation par hôpital

### Concepts clés

**1. Reproduire une référence à l'identique.**
DenseNet-121 sans pré-entraînement, SGD (lr 1e-3, momentum 0,9, weight decay 1e-2), batch 32, 10 époques, aucune
augmentation, époque choisie sur `val` : c'est le protocole ERM de WILDS, relu dans leur code. Reproduire la référence
avant d'innover permet de comparer avec la littérature : sur 4 graines, on obtient 66,1 % ± 14,9 sur le test, contre
70,3 % ± 6,4 dans l'article WILDS (`results/tables/summary_cnn_none.csv`).

**2. L'anti-triche par construction.**
`train.py` ne charge jamais le test. `evaluate.py`, lancé une seule fois sur le modèle retenu, fixe le seuil de décision
sur `val` puis l'applique tel quel au test. Une règle qu'on n'a pas besoin de « respecter » parce que le code la rend
impossible à enfreindre est plus fiable qu'une règle de discipline.

**3. Mesurer avant d'optimiser : trouver le goulot.**
Le banc d'essai sépare chargement des données, calcul et bout en bout. Sur un T4, le chargement n'était pas le goulot
(4,8 fois de marge) ; le calcul était limité par le **lancement** de milliers de petites opérations par le CPU
(petites images, batch de 32). Conséquence contre-intuitive : la précision mixte (`amp`) était 10 % **plus lente**.
Ce qui a vraiment payé : deux graines en parallèle, une par GPU (le quota Kaggle compte le temps de session, pas les GPU).

**4. Reproductibilité : ce qu'on contrôle et ce qu'on ne contrôle pas.**
On contrôle : la graine (initialisation, ordre des batchs), l'ordre des batchs fixé par (graine, époque) pour une
reprise exacte, les points de sauvegarde écrits de façon atomique, le commit du code enregistré dans chaque run.
On ne contrôle pas : certaines opérations GPU (cuDNN) ne sont pas déterministes. Deux exécutions de la même graine ont
divergé dès la première époque. L'**évaluation**, elle, est reproductible : le même point de sauvegarde donne les
mêmes chiffres à 6 décimales sur Kaggle et sur le Mac. D'où : plusieurs graines, et toutes leurs valeurs publiées.

**5. Écart-type d'échantillon.**
Avec 4 graines, on divise par n − 1 (estimateur de l'écart-type d'une population à partir d'un échantillon).
pandas le fait par défaut, NumPy non : piège classique. Avec si peu de valeurs, on publie aussi chaque valeur.

**6. Sélectionner sur un hôpital, c'est s'adapter à cet hôpital.**
Les hôpitaux décalent les probabilités du modèle dans des **sens opposés** : sur l'hôpital 1 (`val`), les modèles retenus
prédisent « tumeur » pour 52 à 57 % des patches (vraie proportion : 50 %) ; sur l'hôpital 2 (test), pour 1 à 38 %.
Choisir l'époque qui maximise `val`, c'est donc favoriser les époques où le modèle penche vers « non tumoral », ce qui
compense l'hôpital 1… et aggrave l'hôpital 2. Sur les 40 points (graine, époque), la corrélation entre précision `val` et
précision `id_val` est de −0,41 ; pour la graine 3, l'époque retenue est celle où `id_val` s'effondre à 64,5 %.
WILDS observe la même chose : sélectionner sur `val` donne 93,2 % ± 5,2 en distribution, contre 98,7 % ± 0,1 en
sélectionnant sur `id_val`. C'est une interprétation cohérente avec les chiffres, pas une preuve causale.
Sources : `results/tables/summary_cnn_none.csv` (métrique `predicted_tumor_share`) et
`results/tables/selection_cnn_none.json`.

**7. AUC et précision ne disent pas la même chose.**
L'AUC mesure si le modèle **classe** correctement (un patch tumoral a-t-il une probabilité plus haute qu'un patch sain ?),
indépendamment du seuil. La précision mesure les décisions **à un seuil donné** (0,5). Sur le test, 3 graines sur 4 ont
une AUC de 0,82 à 0,92 mais une précision de 58 à 82 % : une grande partie de l'échec vient d'un **décalage des
probabilités**, pas seulement d'un mauvais classement. C'est un problème de calibration (phase 5). La graine 0
(AUC 0,62) échoue aussi sur le classement lui-même.

**8. Un seuil ne voyage pas.**
Le seuil réglé sur `val` pour 95 % de spécificité donne entre 0 et 50 % de sensibilité sur le test. Un seuil clinique
réglé dans un hôpital n'est pas valable dans un autre sans recalibration.

### Questions d'entretien

**Q1. Votre modèle fait 88 % sur l'hôpital de validation et 66 % sur l'hôpital de test. Pourquoi, et pourquoi
continuer à sélectionner sur la validation ?**
*(EN: Your model scores 88% on the validation hospital and 66% on the test hospital. Why, and why keep selecting on validation?)*

> Chaque hôpital décale les probabilités du modèle différemment ; ici, l'hôpital de validation pousse vers « tumeur » et
> celui de test vers « non tumoral ». En choisissant l'époque sur la validation, on choisit des modèles adaptés à cet
> hôpital-là : les scores de validation sont serrés (88,2 % ± 0,7) alors que le test varie de 49 à 82 %. Mais c'est le
> protocole officiel, et la seule option honnête sans regarder le test. Les parades : valider sur plusieurs hôpitaux,
> réduire le décalage (normalisation de couleur, phase 4) et recalibrer ou s'abstenir (phase 5).

**Q2. La précision mixte a ralenti votre entraînement. Comment est-ce possible ?**
*(EN: Mixed precision made your training slower. How is that possible?)*

> Avec des images 96×96 et des batchs de 32, chaque opération du DenseNet est minuscule : le GPU attend que le CPU lui
> envoie la suivante. Le temps est dominé par le lancement des opérations, pas par le calcul. La précision mixte ajoute
> des conversions et des contrôles du GradScaler, donc plus de lancements, sans gain de calcul. Preuve : en inférence,
> des batchs de 256 vont 3,7 fois plus vite par image. J'ai mesuré avant d'optimiser, puis utilisé les deux GPU en
> parallèle pour deux graines, ce qui a réellement réduit le coût.

**Q3. Comment rendez-vous un entraînement GPU reproductible ?**
*(EN: How do you make GPU training reproducible?)*

> Graine fixée et passée en paramètre, ordre des batchs déterminé par la graine et l'époque (une reprise après
> interruption redonne exactement les mêmes batchs, vérifié au bit près sur CPU), points de sauvegarde atomiques,
> commit du code et versions enregistrés dans chaque run. Mais certaines opérations GPU ne sont pas déterministes :
> deux exécutions de la même graine ont divergé. La reproductibilité est donc statistique : plusieurs graines, moyenne
> ± écart-type et toutes les valeurs. L'évaluation, elle, est identique à 6 décimales entre Kaggle et mon Mac.

---

## Phase 3 — Caractéristiques figées et classifieurs linéaires

### Concepts clés

**1. Modèle de fondation, caractéristiques figées, *linear probe*.**
Un modèle de fondation est un grand réseau pré-entraîné sur des millions d'images, sans étiquettes (apprentissage
auto-supervisé : DINO, iBOT). On le **gèle** : chaque patch devient un vecteur de 768 nombres (le jeton `[CLS]`), calculé
une seule fois. On n'entraîne qu'une couche linéaire au-dessus (768 × 2 + 2 = 1 538 paramètres, contre environ 7 millions
pour le DenseNet). C'est le *linear probe* : il mesure ce que les caractéristiques contiennent déjà.

**2. Le résultat central.**
Sur l'hôpital de test, avec le même protocole que le CNN (SGD, 10 époques, époque et seuil choisis sur `val`) :
CNN entraîné de zéro 66,1 % ± 14,9 ; ViT figé ImageNet (DINO) 92,1 % ± 0,3 ; ViT figé histologie (Phikon) 93,9 % ± 0,8
(`results/tables/comparison.csv`). La variance entre graines s'effondre : le problème linéaire est convexe, il n'y a
presque plus d'instabilité d'entraînement.

**3. Pourquoi ça généralise mieux (interprétation).**
Un CNN entraîné de zéro sur 30 lames apprend tout ce qui sépare les classes dans les hôpitaux d'entraînement, y compris
les indices propres à ces hôpitaux (coloration, scanner) : il a des millions de paramètres libres pour les exploiter.
Un réseau pré-entraîné sur des données très variées fournit des caractéristiques plus générales (formes, textures,
noyaux), et une couche linéaire de 1 538 paramètres a peu de liberté pour s'accrocher à des indices subtils. Les
probabilités sont aussi bien plus stables d'un hôpital à l'autre : part de patches déclarés « tumeur » entre 45 et 55 %
partout (vraie part 50 %), contre 1 à 57 % pour le CNN.

**4. Ce que la comparaison ne permet pas de séparer.**
CNN contre têtes linéaires mélange trois différences : l'architecture (DenseNet contre ViT), le pré-entraînement et le
gel des poids. Pour les séparer, il faudrait par exemple affiner (*fine-tuner*) un ViT pré-entraîné, ou entraîner un ViT
de zéro. En revanche, **DINO contre Phikon isole les données de pré-entraînement** : même architecture, même famille
d'apprentissage auto-supervisé, même prétraitement.

**5. Histologie contre ImageNet : un avantage net à la maison, mince sur l'hôpital difficile.**
Phikon fait mieux en distribution (99,0 % contre 97,4 %) et sur l'hôpital 1 (97,1 % contre 92,0 %). Sur l'hôpital 2, la
précision diffère de 1,8 point (93,9 contre 92,1), mais **l'AUC est identique (0,986)** : les deux classent aussi bien,
l'écart vient du seuil (Phikon déclare « tumeur » pour 45,5 % des patches, DINO pour 55,2 %). L'essentiel du gain par
rapport au CNN vient donc du pré-entraînement à grande échelle et du gel, pas spécifiquement de l'histologie.

**6. Ce que les graines mesurent, et ce qu'elles ne mesurent pas.**
Pour une tête linéaire, la graine ne change que l'initialisation et l'ordre des batchs : écart-type inférieur à 1 point.
Elle ne mesure pas l'incertitude liée au **choix des patients** dans l'hôpital de test (10 lames seulement). Pour affirmer
que Phikon bat DINO sur le test, il faudrait un intervalle de confiance par rééchantillonnage des lames ou des patients
(*bootstrap*) : c'est une limite, à traiter en phase 5.

**7. Vérifier une extraction de caractéristiques.**
Versions des modèles épinglées ; prétraitement identique à celui du modèle (écart de 2,4e-7 avec le préprocesseur
Hugging Face) ; fp16 contre fp32 (similarité cosinus ≥ 0,99998) ; lignes réalignées par les index sauvegardés ;
effectifs comparés à l'article ; commit du code dans `meta.json`. Et la question de la **contamination** : Phikon a été
pré-entraîné sur TCGA, pas sur Camelyon, donc pas de fuite connue vers notre test.

**8. Le piège des axes automatiques.**
Sur la figure des courbes de Phikon, `val` varie entre 96,4 et 97,1 % : l'axe automatique fait paraître ces variations
aussi fortes que celles du CNN (74 à 89 %), vingt fois plus grandes. Pour comparer des méthodes, une même échelle pour
toutes (`results/figures/comparison.png`).

### Questions d'entretien

**Q1. Pourquoi un modèle pré-entraîné figé avec une couche linéaire généralise-t-il mieux à un nouvel hôpital qu'un CNN
entraîné de bout en bout ?**
*(EN: Why does a frozen pretrained model with a linear head generalize better to a new hospital than an end-to-end CNN?)*

> Le CNN, entraîné de zéro sur 30 lames, a des millions de paramètres pour exploiter tout ce qui sépare les classes dans
> les hôpitaux d'entraînement, y compris la coloration propre à ces hôpitaux. Le réseau figé a appris sur des millions
> d'images variées des caractéristiques plus générales, et la tête linéaire n'a que 1 538 paramètres. Résultat sur
> l'hôpital de test : 92 à 94 % contre 66 %, et un écart-type entre graines inférieur à 1 point contre 15. Limite : ma
> comparaison mélange architecture, pré-entraînement et gel ; il faudrait affiner un ViT pré-entraîné pour les séparer.

**Q2. Un modèle de fondation spécialisé en histologie est-il meilleur qu'un modèle ImageNet ici ?**
*(EN: Is a histology foundation model better than an ImageNet model here?)*

> En distribution et sur l'hôpital de validation, oui : 99,0 % contre 97,4 %, et 97,1 % contre 92,0 %. Sur l'hôpital de
> test, l'AUC est identique (0,986) et la précision ne diffère que de 1,8 point, écart qui vient du seuil de décision,
> pas de la qualité du classement. Mes graines ne mesurent que l'aléa d'entraînement, pas le choix des patients de test ;
> sans intervalle de confiance par bootstrap sur les lames, je ne conclus pas à une différence sur le test.

**Q3. Comment avez-vous vérifié que l'extraction des caractéristiques était correcte ?**
*(EN: How did you check that the feature extraction was correct?)*

> Versions des modèles épinglées ; prétraitement comparé au préprocesseur officiel (écart de 2,4e-7) ; demi-précision
> comparée à la pleine précision (similarité cosinus ≥ 0,99998) ; chaque ligne réalignée sur son étiquette grâce aux
> index sauvegardés ; effectifs comparés à l'article ; commit du code enregistré. J'ai aussi vérifié que Phikon n'avait
> pas été pré-entraîné sur Camelyon, pour écarter une contamination du test.

---

## Phase 4 — Couleur : augmentation HED et normalisation de Macenko

### Concepts clés

**1. Deux réponses au décalage de couleur : apprendre à l'ignorer, ou l'effacer.**
L'**augmentation** (HED jitter) modifie au hasard la coloration de chaque patch, **à l'entraînement seulement** : le
modèle voit plus de colorations que les trois hôpitaux d'entraînement n'en offrent, et doit s'appuyer sur autre chose
(formes, textures). À l'inférence, pas de hasard : sinon la même image recevrait une prédiction différente à chaque appel.
La **normalisation** (Macenko) ramène chaque image vers une coloration de référence, **à l'entraînement comme à
l'inférence** (`val`, test, et plus tard l'application) : le modèle n'a vu que des images normalisées, lui donner une
image brute serait un nouveau décalage. Un test vérifie cette règle pour chaque configuration (`tests/test_transforms.py`).

**2. Pourquoi travailler en densité optique plutôt qu'en RGB.**
En densité optique (phase 1), un pixel est la somme d'une contribution d'hématoxyline et d'une contribution d'éosine,
chacune proportionnelle à la quantité de colorant. « Un peu plus d'éosine » devient une opération simple : multiplier
une concentration. Alternative écartée : `ColorJitter` de torchvision (luminosité, contraste, teinte en RGB), qui
produit des couleurs qu'aucune coloration H&E ne donne et ne sépare pas les deux colorants.

**3. HED jitter (Tellez et al., 2019).**
Chaque pixel est décomposé en trois concentrations : H (hématoxyline), E (éosine) et D (DAB, troisième canal de la
matrice, presque vide en H&E). Pour chaque canal, on tire α ~ U(1 − σ, 1 + σ) et β ~ U(−σ, σ), on applique
C → α·C + β à tous les pixels du patch, puis on reconstruit l'image. σ = 0,05 (« HED-light » ; Tellez appelle
« HED-strong » σ = 0,2). Cette valeur a été fixée **avant tout entraînement**, sur une vérification visuelle
(`results/figures/hed_light_examples.png`), et n'a pas été réglée : la régler se ferait sur `val`, jamais sur le test.
Piège d'unités : `rgb2hed` de scikit-image divise l'OD par 13,8 (= −log(1e-6)) ; dans ces unités, σ = 0,05 serait
environ 14 fois plus fort. Notre code applique la matrice `hed_from_rgb` directement à l'OD (logarithme naturel).

**4. La normalisation de Macenko, étape par étape (Macenko et al., 2009).**
Pour chaque patch :
1. Garder les pixels de tissu (norme de l'OD ≥ 0,15).
2. Dans l'espace OD à trois dimensions, ces pixels forment un nuage presque plat, puisque chacun est un mélange de deux
   colorants : on trouve ce plan par analyse en composantes principales (les deux directions principales).
3. Mesurer l'angle de chaque pixel dans ce plan ; les deux angles extrêmes (1er et 99e centiles, plus robustes que le
   minimum et le maximum) donnent les deux colorants purs. L'hématoxyline est celui qui absorbe le plus le rouge.
4. Calculer la concentration de chaque colorant par moindres carrés.
5. Mettre à l'échelle (99e centile de chaque concentration → valeur de référence), puis reconstruire l'image avec les
   vecteurs de colorant de la référence.

Si le patch a moins de 5 % de tissu, ou si le calcul échoue, on garde l'image d'origine (repli). Taux de repli mesuré :
0 sur 999 patches de train, 0 sur 999 patches de `val` (`results/tables/macenko_fallback.json`).

**5. Une référence calculée sur le train uniquement, et appliquée patch par patch.**
La référence est la médiane des vecteurs de colorant et des concentrations maximales de 999 patches de train, répartis
entre les hôpitaux 0, 3 et 4 (`results/tables/macenko_reference.json`). Utiliser des images de l'hôpital cible, même
sans leurs étiquettes, serait de l'**adaptation de domaine** : un autre problème que celui de WILDS, qui mesure la
généralisation à un hôpital jamais vu. Chaque patch est normalisé seul : l'application pourra traiter une image isolée
d'un hôpital inconnu. Contrepartie : une estimation sur 96 × 96 pixels est plus bruitée que sur une lame entière.

**6. Un écart volontaire avec les implémentations courantes.**
Beaucoup d'implémentations de Macenko gardent un pixel si son OD dépasse 0,15 **dans chaque canal**. Or l'éosine absorbe
peu le rouge : un pixel de stroma rose a une OD rouge proche de 0,1. Ces pixels sont exclus, et le vecteur d'éosine
estimé est biaisé. Nous seuillons la norme de l'OD. Un test sur des pixels synthétiques de colorants connus, dont 10 %
d'éosine pure, vérifie que les deux vecteurs sont retrouvés (similarité cosinus > 0,999, `tests/test_color.py`).

**7. Résultat de l'augmentation HED : moins d'échecs graves, pas encore la solution.**
Quatre graines, même protocole que la référence, époque et seuil choisis sur `val` (`results/tables/comparison.csv`) :

| Mesure | Sans couleur | HED-light |
|---|---|---|
| Précision, hôpital de test (2) | 66,1 % ± 14,9 | 81,0 % ± 5,1 |
| Pire graine, hôpital de test | 49,4 % | 77,1 % |
| AUC, hôpital de test | 0,817 | 0,918 |
| Sensibilité au seuil choisi sur `val`, hôpital de test | 17,3 % | 46,2 % |
| Part déclarée « tumeur » sur le test (vraie part : 50 %) | 18,7 % | 36,1 % |
| Précision en distribution (`id_val`) | 87,2 % ± 15,2 | 96,5 % ± 1,3 |

L'AUC augmente de 10 points : le **classement** s'améliore, pas seulement le seuil. Autre effet : sur l'ensemble des
époques, la corrélation entre précision sur `val` et sur `id_val` passe de −0,41 à +0,52
(`results/tables/selection_cnn_*.json`). Choisir l'époque sur l'hôpital 1 ne sacrifie plus les hôpitaux d'entraînement :
ils ne tirent plus les probabilités dans des sens opposés. Mais le modèle sous-détecte encore sur l'hôpital 2 (36 % de
patches déclarés tumeur pour 50 %), et reste loin des têtes linéaires sur caractéristiques figées (92 à 94 %).

**8. Quatre graines : la différence est-elle significative ?**
Test t de Welch (il ne suppose pas des variances égales, ici 14,9 contre 5,1 points) sur les quatre précisions par
graine (`results/tables/summary_cnn_none.csv` et `summary_cnn_hed.csv`) : t = 1,89, environ 3,7 degrés de liberté,
**p = 0,14**. Non significatif au seuil de 5 %. Cela ne veut pas dire « aucun effet » : avec quatre graines et un
écart-type de 15 points pour la référence, le test a peu de puissance. On peut affirmer honnêtement : gain moyen de
15 points, non significatif avec quatre graines ; écart-type divisé par trois ; la pire graine HED (77,1 %) dépasse
trois des quatre graines de la référence. Le nombre de graines a été fixé **avant** de voir les résultats : en ajouter
jusqu'à obtenir p < 0,05 serait du *p-hacking*. Et comme en phase 3, les graines ne mesurent pas l'aléa du choix des
patients de test (bootstrap par lame, phase 5).

**9. Le coût d'une transformation à la volée.**
Les transformations de couleur tournent sur le processeur, dans les *workers* du DataLoader, à chaque époque. Durée
d'entraînement de 10 époques (`train_minutes` dans `train_summary.json`) : 143 à 145 min sans couleur, 180 à 202 min
avec HED. Macenko (une analyse en composantes principales et des moindres carrés par patch) n'avait pas fini sa première
époque après 45 min au premier essai sur Kaggle. Deux corrections, mesurées avant d'être adoptées :
- `pinv` au lieu de `lstsq` pour les moindres carrés : 44 % plus rapide par patch, images identiques au pixel près sur
  600 patches ;
- un seul fil de calcul BLAS par *worker* : sinon chaque *worker* lance un fil NumPy par cœur, et les *workers* des deux
  entraînements parallèles se disputent les quelques cœurs de la machine (*oversubscription*).

Alternative écartée : pré-calculer une fois les images normalisées (plus rapide à l'entraînement, mais une copie du jeu
de données et du code en plus). Effet des corrections : 25,6 à 28,8 min par époque sur la relance (`epoch_minutes` dans
`history.csv`), contre plus de 45 min avant.

**10. Un cas limite mesuré : le débordement numérique.**
Les journaux Kaggle de Macenko contiennent 60 avertissements `overflow encountered in exp` par graine. Python n'affiche
un avertissement qu'une fois par processus : 60 = 2 *workers* × 3 chargeurs (train, `val`, `id_val`) × 10 époques, car
les *workers* sont recréés à chaque passage. Le nombre de lignes ne dit donc rien de la fréquence ; il faut la mesurer
(`python -m src.color check --n-patches 9999`, `results/tables/macenko_checks.json`) : 3 patches sur 9 999 en train,
en `id_val` et en `val` (0,03 %), aucun sur 9 999 patches de test. Tous ont au moins 44 % de pixels saturés (un canal
à 0, donc une OD à son plafond) : leurs concentrations sont extrêmes, certaines OD reconstruites sont très négatives,
`exp(-OD)` déborde et `np.clip` ramène ces valeurs à la luminosité maximale. L'image normalisée est quand même produite.
Décision : code inchangé pour les graines 2 et 3, car les quatre graines doivent utiliser exactement la même méthode.
Une correction (exclure ou plafonner les pixels saturés) changerait la méthode et obligerait à réentraîner les quatre
graines ; c'est une limite documentée.

### Résultats Macenko et questions d'entretien

À compléter quand les quatre graines `cnn_macenko` seront terminées : comparaison des trois conditions de couleur,
puis trois questions d'entretien avec leurs réponses.
