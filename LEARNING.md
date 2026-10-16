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
