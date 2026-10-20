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

### Résultats : les trois conditions de couleur

**11. Macenko : un CNN entraîné de zéro rejoint les modèles de fondation sur l'hôpital 2.**
Quatre graines par condition, époque et seuil choisis sur `val` (`results/tables/comparison.csv`,
`results/tables/summary_cnn_*.csv`) :

| Mesure | Sans couleur | HED-light | Macenko |
|---|---|---|---|
| Précision, hôpital de test (2) | 66,1 % ± 14,9 | 81,0 % ± 5,1 | 93,5 % ± 1,1 |
| Pire graine, hôpital de test | 49,4 % | 77,1 % | 91,8 % |
| AUC, hôpital de test | 0,817 | 0,918 | 0,986 |
| Sensibilité au seuil choisi sur `val`, hôpital de test | 17,3 % | 46,2 % | 96,0 % |
| Part déclarée « tumeur » sur le test (vraie part : 50 %) | 18,7 % | 36,1 % | 51,5 % |
| Précision sur `val` (hôpital 1) | 88,2 % ± 0,7 | 89,9 % ± 1,0 | 89,2 % ± 1,6 |
| Précision en distribution (`id_val`) | 87,2 % ± 15,2 | 96,5 % ± 1,3 | 96,1 % ± 1,3 |

Sur l'hôpital 2, le CNN normalisé atteint le niveau des têtes linéaires sur caractéristiques figées (DINO 92,1 %,
Phikon 93,9 %), avec la même AUC (0,986), et le biais de seuil disparaît (51,5 % de patches déclarés tumeur pour 50 %).
Pour cet hôpital, l'obstacle principal du CNN était donc la couleur. Hypothèse sur l'écart avec HED : la normalisation
supprime le décalage au moment de la prédiction, alors que l'augmentation légère (± 5 %) élargit peu la distribution
d'entraînement ; HED-strong (σ = 0,2) n'a pas été testé, c'est une limite.

**12. Ce que la normalisation ne règle pas : l'hôpital 1.**
Sur `val`, Macenko (89,2 %) fait à peine mieux que le CNN sans couleur (88,2 %), alors que Phikon atteint 97,1 %. La
difficulté de l'hôpital 1 n'est donc probablement pas (seulement) la couleur : netteté, épaisseur des coupes, types de
tissu ? C'est une question ouverte pour les galeries d'erreurs de la phase 5. Et la précision sur `val` varie beaucoup
d'une époque à l'autre (de 69 à 91 % selon les graines, `results/figures/curves_cnn_macenko.png`) : l'époque choisie
dépend en partie du hasard.

**13. `val` ne classe pas les méthodes comme le test.**
Sur `val`, HED devance Macenko de 0,7 point ; sur le test, Macenko gagne de 12,5 points. Si l'on avait dû choisir la
méthode de couleur sur `val`, on aurait pris HED, à un écart qui relève du bruit. Un seul hôpital de validation est un
guide faible pour choisir entre méthodes. Conséquences : on rapporte **toutes** les méthodes, sans en sélectionner une
d'après le test ; et pour choisir le modèle de l'application (phase 6), il faudra un critère défini sur `val` ou fixé à
l'avance, jamais « le meilleur sur le test ». Une piste plus solide pour un vrai projet : plusieurs hôpitaux de
validation (validation croisée en laissant un hôpital de côté à chaque fois).

**14. Significativité : quatre comparaisons, prudence.**
Tests t de Welch sur les quatre précisions par graine sur le test (calculés à partir des valeurs des fichiers
`summary_*.csv`) : Macenko contre sans couleur p = 0,035 ; contre HED p = 0,014 ; contre DINO p = 0,085 ; contre Phikon
p = 0,61. Avec quatre tests, la correction de Bonferroni demande p < 0,05 / 4 = 0,0125 : aucun ne passe strictement.
Un argument plus simple : les graines ne se chevauchent pas (pire graine Macenko 91,8 %, meilleure graine HED 88,5 %,
meilleure graine sans couleur 81,9 %). Conclusion honnête : Macenko dépasse nettement HED et l'absence de traitement
sur l'hôpital 2 ; on ne peut pas le départager des modèles de fondation. Toujours avec la limite des graines : elles ne
mesurent pas le choix des patients de test (bootstrap par lame en phase 5).

**15. Le prix de la normalisation à la volée.**
Durée d'entraînement de 10 époques sur une T4 (`train_minutes`) : environ 144 min sans couleur, 180 à 202 avec HED,
278 à 283 avec Macenko, soit environ deux fois plus que sans couleur. Et la même normalisation, avec la même référence,
devra être appliquée à chaque image dans l'application.

### Questions d'entretien

**Q1. Augmentation de couleur ou normalisation de couleur : laquelle avez-vous retenue, et pourquoi ?**
*(EN: Stain augmentation or stain normalization: which one did you pick, and why?)*

> J'ai testé les deux, avec quatre graines chacune et le même protocole. L'augmentation HED, appliquée seulement à
> l'entraînement, fait passer l'hôpital de test de 66 à 81 % ; la normalisation de Macenko, appliquée à chaque image à
> l'entraînement comme à l'inférence, l'amène à 93,5 %, au niveau des modèles de fondation figés, avec un écart-type
> de 1 point au lieu de 15. Mais elle n'améliore presque pas l'hôpital de validation (89 % contre 88 %), et elle
> double le temps d'entraînement.
> Ce n'est pas un choix universel : c'est le résultat sur cet hôpital, avec une seule intensité d'augmentation testée.

**Q2. Votre hôpital de validation classe HED devant Macenko, le test dit l'inverse. Que faites-vous ?**
*(EN: Your validation hospital ranks HED above Macenko, but the test says the opposite. What do you do?)*

> Je ne change pas de méthode d'après le test : sinon le test cesse de mesurer la généralisation, il devient un second
> jeu de validation. Je rapporte les deux résultats, et je constate qu'un seul hôpital de validation est un mauvais
> guide pour choisir entre méthodes (0,7 point d'écart sur `val`, 12,5 sur le test). Pour un vrai déploiement,
> j'utiliserais plusieurs hôpitaux de validation, en laissant un hôpital de côté à chaque fois, et je fixerais le
> critère de choix avant de regarder le test.

**Q3. Comment vous assurez-vous que la normalisation ne fait pas fuiter d'information de l'hôpital de test ?**
*(EN: How do you make sure stain normalization does not leak information from the test hospital?)*

> La référence de couleur est calculée sur 999 patches d'entraînement seulement, puis figée dans un fichier versionné.
> Chaque patch est normalisé seul, sans statistique calculée sur d'autres images de son hôpital : c'est exactement ce
> que ferait l'application face à une image d'un hôpital inconnu. Utiliser des images du test, même sans étiquettes,
> serait de l'adaptation de domaine, un autre problème. Les seules mesures faites sur les couleurs du test (taux de
> repli et de débordement) l'ont été après avoir figé la méthode, et ne servent à rien choisir.

---

## Phase 5 — Calibration, abstention, analyse des erreurs

### Concepts clés

**1. Calibration : la probabilité dit-elle vrai ?**
Un modèle est calibré si, parmi les patches à qui il donne environ 80 % de probabilité de tumeur, environ 80 % sont des
tumeurs. Trois mesures :
- le **diagramme de fiabilité** : on découpe les probabilités en 15 tranches de même largeur et on compare, dans chaque
  tranche, la probabilité moyenne à la part réelle de tumeurs (sur la diagonale : calibré) ;
- l'**ECE** (*expected calibration error*) : l'écart moyen entre les deux, pondéré par le nombre de patches par tranche ;
- le **score de Brier** : l'erreur quadratique moyenne de la probabilité, sans découpage en tranches.

Précision et calibration sont deux qualités différentes : un modèle peut bien classer et annoncer des probabilités
fausses. Pour un outil qui affiche une confiance à un médecin, la calibration compte autant que la précision.

**2. Le *temperature scaling* (Guo et al., 2017).**
On divise le logit par un seul nombre T, ajusté sur `val` en minimisant la log-loss. T > 1 adoucit un modèle trop sûr
de lui, T < 1 durcit un modèle trop timide. Comme T ne change ni l'ordre des patches ni le signe du logit, l'AUC, les
décisions au seuil 0,5 et le seuil choisi sur `val` restent identiques : toutes les métriques déjà publiées restent
valables. Températures obtenues (`results/tables/calibration.csv`) : 1,3 à 1,5 pour les CNN (trop sûrs d'eux sur
l'hôpital 1), 1,13 pour DINO, 0,85 pour Phikon (trop timide). Alternatives écartées : la méthode de Platt (pente et
biais : le biais déplacerait les décisions) et la régression isotonique (non paramétrique, elle sur-apprend).

**3. Résultat : une température apprise sur un hôpital ne se transporte pas.**
ECE sur l'hôpital de test, avant → après la température ajustée sur l'hôpital 1 (moyenne sur 4 graines) :
CNN sans couleur 0,294 → 0,261 ; HED 0,126 → 0,113 ; Macenko 0,046 → 0,067 ; DINO 0,070 → 0,073 ;
Phikon 0,039 → 0,040. Sur les hôpitaux d'entraînement, la correction dégrade les CNN (HED 0,031 → 0,055 ; Macenko
0,034 → 0,064). Explication (`results/figures/reliability.png`) : d'un hôpital à l'autre, l'erreur de calibration est
surtout un **décalage** dans une direction propre à l'hôpital. Sur l'hôpital 2, les courbes du CNN sans couleur, de HED
et de Phikon passent au-dessus de la diagonale (ils sous-estiment la tumeur), celles de DINO et de Macenko en dessous
(ils la surestiment). Une température change l'échelle des probabilités, pas leur direction : elle ne peut pas
corriger des décalages opposés. **La calibration est une propriété du couple (modèle, hôpital), pas du modèle seul.**

**4. Conséquence pour l'application.**
La probabilité affichée ne sera pas fiable sur un hôpital inconnu, même après correction ; on ne peut d'ailleurs pas
vérifier une correction sans un troisième hôpital non vu. Le README devra le dire. Le seuil « à revoir par un
pathologiste » doit donc s'appuyer sur ce qui se transporte mieux : le classement (AUC) et la courbe risque–couverture.

**5. Les pièges de l'ECE.**
Elle dépend du découpage en tranches, et peut contredire le score de Brier : sur `id_val`, la correction fait passer
l'ECE du CNN sans couleur de 0,126 à 0,150 alors que son Brier s'améliore (0,104 → 0,100). D'où les deux mesures. Le
code a été vérifié de façon indépendante sur un run : ECE recalculée par une boucle explicite et température obtenue
par une régression logistique de scikit-learn, identiques.

**6. Intervalles de confiance par bootstrap sur les patients.**
Chaque hôpital inconnu, ce sont 10 lames de 8 patients (`val`) ou de 9 patients (test). Un autre hôpital, avec d'autres
patients, donnerait d'autres chiffres. Le bootstrap rejoue ce hasard : 2 000 fois, on re-tire les patients **avec
remise** (certains deux fois, d'autres pas du tout), on garde tous leurs patches et on recalcule la métrique ; les 2,5e
et 97,5e centiles donnent l'intervalle à 95 % (`python -m src.bootstrap`, `results/tables/bootstrap.csv`).
- Des patients, pas des patches : les patches d'une lame partagent tissu et coloration, ils ne sont pas indépendants.
  Un bootstrap sur 85 000 patches donnerait des intervalles absurdement étroits.
- Des patients, pas des lames : deux lames d'un même patient partagent sa biologie (le patient 046 a deux lames dans
  l'hôpital 2), même raisonnement que la validation croisée par patient de la phase 1.
- Tirages **appariés** : toutes les méthodes reçoivent les mêmes tirages, ce qui donne aussi un intervalle sur leurs
  différences (`results/tables/bootstrap_differences.csv`).
- Chaque tirage utilise la moyenne des 4 graines : l'intervalle mesure le choix des patients, l'écart-type entre graines
  mesure l'aléa d'entraînement ; ce sont deux sources d'incertitude différentes.

**7. Ce que disent les intervalles.**
Précision sur l'hôpital de test, avec intervalle à 95 % : sans couleur 66,1 [56,4 ; 86,0] ; HED 81,0 [73,7 ; 88,9] ;
Macenko 93,5 [91,4 ; 94,9] ; DINO 92,1 [85,7 ; 95,5] ; Phikon 93,9 [84,0 ; 98,0] (`results/figures/bootstrap.png`).
Différences appariées :
- Macenko devance HED de 12,5 points [2,9 ; 19,3] : HED fait mieux dans 0,25 % des tirages seulement. Ce gain-là
  résiste au choix des patients.
- Les trois meilleurs sont **indiscernables** sur le test : Macenko − Phikon −0,4 [−6,1 ; +8,6], Macenko − DINO
  +1,4 [−0,7 ; +5,6], DINO − Phikon −1,8 [−10,7 ; +7,1] ; leurs AUC diffèrent de moins de 0,001 en moyenne.
- Sur `val`, au contraire, Phikon devance tous les autres (par exemple de 7,9 points [5,9 ; 9,0] sur Macenko), et HED
  et Macenko sont indiscernables (+0,7 [−3,6 ; +4,0]).

**8. Un patient pèse lourd.**
La lame 28 (patient 051) contient 37 % des patches de l'hôpital de test et 63 % de ses patches tumoraux
(`results/tables/slides.csv`). Une précision calculée sur tous les patches mélangés dépend donc beaucoup de ce
patient : c'est pourquoi certains intervalles sont si larges (CNN sans couleur : 56 à 86 %). Un vrai rapport clinique
donnerait aussi les résultats patient par patient.

**9. Les limites de l'outil.**
Avec 8 ou 9 patients, l'intervalle par centiles est approximatif et tend à être **trop étroit** : il est indicatif.
Surtout, il mesure la variabilité des patients **à l'intérieur** d'un hôpital, pas d'un hôpital à l'autre : avec deux
hôpitaux inconnus seulement, on ne sait pas comment ces méthodes se comporteraient sur un troisième.

**10. Abstention : la courbe risque–couverture.**
Le modèle peut **ne pas répondre** et renvoyer un patch à un pathologiste. On classe les patches du plus sûr au moins
sûr (confiance : distance de la probabilité à 0,5) ; la **couverture** est la part de patches auxquels le modèle
répond, le **risque** le taux d'erreur parmi eux. Si la confiance est utile, le risque baisse quand on renvoie les
patches les moins sûrs (`python -m src.abstention`, `results/figures/risk_coverage.png`). Deux résumés : l'**AURC**
(aire sous la courbe, plus bas = mieux) mélange précision et qualité de la confiance ; l'**E-AURC** retire ce que ferait
un oracle qui classerait toutes les erreurs en dernier, et isole la qualité du classement par confiance. La température
ne change pas l'ordre des confiances : ces courbes sont les mêmes avant et après calibration.

**11. Ce que donne un renvoi de 20 % des patches.**
Seuil de confiance choisi sur `val` pour répondre à 80 % de ses patches, appliqué tel quel à l'hôpital de test
(`results/tables/abstention_cutoffs.csv`, moyenne sur 4 graines) : erreur de 6,5 % → 2,2 % pour Macenko (83 % des
patches traités), 7,9 % → 2,3 % pour DINO (79 %), 6,1 % → 1,0 % pour Phikon (70 %), mais 19,0 % → 10,8 % pour HED et
33,9 % → 29,7 % pour le CNN sans couleur. Pour ce dernier, la confiance ne sert presque à rien sur l'hôpital 2 : ses
erreurs sont **sûres d'elles** (E-AURC sur le test 0,116, contre 0,009 à 0,011 pour les trois meilleurs,
`results/tables/risk_coverage.csv`). L'abstention aide un bon modèle ; elle ne sauve pas un modèle biaisé.

**12. Une promesse de couverture se transporte mieux qu'une promesse d'erreur.**
Seuil choisi sur `val` pour « au plus 2 % d'erreur », appliqué à l'hôpital de test : 0,4 % d'erreur pour Macenko (mais
14 % de patches traités seulement), 1,5 % pour DINO, 2,5 % pour HED, **4,3 % pour Phikon** et 6,5 % pour le CNN sans
couleur. La garantie d'erreur obtenue sur un hôpital ne vaut pas ailleurs, même pour le meilleur modèle. La couverture
varie aussi, mais moins : le seuil « 80 % » donne entre 69 et 83 % sur le test selon les méthodes. Conséquences pour
l'application : présenter la règle comme « le modèle renvoie les patches les moins sûrs (seuil réglé sur l'hôpital 1) »,
jamais comme un taux d'erreur garanti ; et prévenir que la charge de relecture du pathologiste change d'un hôpital à
l'autre.

**13. Macenko et l'hôpital 1, encore.**
Sur `val`, Macenko a le moins bon classement par confiance des cinq méthodes (E-AURC 0,037, contre 0,025 pour HED et
0,002 pour Phikon) : même ses patches les plus sûrs contiennent des erreurs. C'est un deuxième indice, après la
précision, que la difficulté de l'hôpital 1 n'est pas la couleur. Les galeries d'erreurs devront regarder ces erreurs
sûres d'elles.

**14. Galeries d'erreurs : regarder les images.**
Graine 0 de chaque méthode, fixée à l'avance ; pour chaque hôpital inconnu, les faux négatifs et faux positifs les plus
sûrs d'eux (`python -m src.explain`, `results/figures/errors_val.png` et `errors_test.png`). Ce sont des cas extrêmes,
pas des erreurs typiques. Ce que l'on voit, à l'œil non expert (à faire confirmer par un pathologiste) :
- Hôpital 2 : les tumeurs manquées par le CNN sans couleur et par HED sont des patches de la lame 28 occupés surtout par
  de la graisse (grandes vacuoles blanches) avec peu de cellules ; celles manquées par Macenko, et par les cinq méthodes
  à la fois, sont des patches de la lame 28 remplis de petits noyaux ronds et denses, qui ressemblent davantage à du
  tissu lymphoïde qu'à un carcinome.
- Hôpital 1 : HED et DINO manquent des zones roses presque sans noyaux de la lame 13, étiquetées tumeur (hypothèse :
  de la nécrose à l'intérieur d'une tumeur).
- Fausses alertes : stroma dense, tissu flou (lame 21 pour le CNN sans couleur), et même, pour DINO, un patch presque
  entièrement vide (lame 25) déclaré tumeur à 100 %.

**15. Les erreurs partagées : des candidats à une étiquette discutable.**
Sur l'hôpital de test, 480 patches sont mal classés par les cinq méthodes, dont 474 tumeurs manquées : 236 viennent de
la lame 28 et 151 de la lame 29 (`results/tables/error_overlap.csv`, `shared_errors_by_slide.csv`). Une erreur commise
avec assurance par cinq modèles très différents vient plus probablement de l'image ou de l'étiquette que d'un modèle :
l'étiquette d'un patch dérive d'un contour tracé à la main, et elle ne décrit que sa région centrale de 32 × 32 pixels.
Ce sont des candidats à faire relire, pas une preuve d'erreur d'étiquette. Ils pèsent peu (0,6 % des patches de test),
mais rappellent que la « vérité » d'un jeu de données médical est elle-même imparfaite.

**16. Grad-CAM : où les CNN regardent.**
Grad-CAM (Selvaraju et al., 2017) pondère chaque carte d'activation de la dernière couche par la moyenne du gradient du
score « tumeur », puis garde la partie positive (`results/figures/gradcam.png`, mêmes patches pour les trois CNN). Sur
l'hôpital 2, le CNN normalisé allume les zones cellulaires et laisse le fond blanc éteint ; le CNN sans couleur, qui
rate ces tumeurs, n'a **aucune** case qui pousse vers « tumeur » (carte nulle), ce qui illustre son biais. Pour les
tumeurs bien détectées, la case centrale porte 14 à 17 % de la carte, à peine plus que les 11 % d'une carte uniforme
(`results/tables/gradcam_center.csv`) : les CNN utilisent tout le patch, pas seulement la région décrite par
l'étiquette, ce qui est plausible puisqu'une zone tumorale dépasse en général 32 pixels.

**17. Les limites de Grad-CAM.**
- Résolution : sur un patch de 96 pixels, la dernière couche de DenseNet est une grille de 3 × 3 cases. On l'affiche
  sans lissage, qui inventerait des détails.
- Chaque case « voit » bien plus que ses 32 pixels : le champ récepteur des couches profondes couvre presque tout le
  patch. La carte dit où se trouvent les activations utiles, pas quels pixels ont compté.
- Chaque carte est ramenée à son propre maximum : on compare des formes, pas des intensités.
- Une carte de saillance n'est pas une explication du raisonnement du modèle. Un test de bon sens classique (Adebayo et
  al., 2018) vérifie que la carte change quand on remplace les poids par des poids aléatoires ; il n'a pas été fait ici.
