# Plan du projet — détection de métastases robuste au changement d'hôpital

> Document de travail (en français). Version du 17 octobre 2026, à valider avant toute ligne de code d'entraînement.
> Objectif : v1 publique (code, README, démo en ligne) **avant le 27 novembre 2026**.

## 1. La question scientifique

Un modèle entraîné sur les lames de 3 hôpitaux garde-t-il ses performances sur un hôpital qu'il n'a jamais vu ?
Et quelles techniques réduisent l'écart : gestion de la couleur, ou caractéristiques d'un modèle pré-entraîné sur de l'histologie (Phikon) ?
On veut aussi savoir **quand le modèle doit s'abstenir** (renvoyer au pathologiste).

## 2. Faits vérifiés en phase 0 (sources : code du package `wilds` 2.0.0, README WILDS)

| Élément | Valeur | Source |
|---|---|---|
| Tâche | patch 96×96 ; y = 1 si la **zone centrale 32×32** contient du tissu tumoral | docstring `Camelyon17Dataset` |
| Hôpitaux | 5, numérotés 0 à 4 dans le code (1 à 5 dans l'article) | `camelyon17_dataset.py` |
| `train` / `id_val` | hôpitaux 0, 3, 4 (découpage fourni dans `metadata.csv`) | idem |
| `val` (OOD) | hôpital 1 | `VAL_CENTER = 1` |
| `test` (OOD) | hôpital 2 | `TEST_CENTER = 2` |
| Évaluateur officiel | précision moyenne, groupes = **lames** (pas hôpitaux) | `Camelyon17Dataset.eval` |
| Protocole de référence | DenseNet-121 **sans pré-entraînement**, SGD (momentum 0,9), lr 1e-3, weight decay 1e-2, batch 32, 10 époques, sélection sur `val` | `examples/configs/datasets.py` |
| Taille | archive 10,7 Go, 15 Go sur disque | code + README WILDS |
| Coût de référence | ~2 h par entraînement complet sur GPU V100 | README WILDS |
| Variance | les auteurs préviennent : forte variabilité du test selon la graine | README WILDS |

**Dépendances** : `torch-scatter` / `torch-geometric` sont listés dans le README WILDS mais **ne sont pas déclarés** par le paquet (vérifié sur PyPI). `pip install wilds` ne les installe pas, et ils ne servent qu'aux jeux de données de graphes. Testé : `wilds` 2.0.0 s'installe et s'importe avec torch 2.14, numpy 2.5, pandas 3.0 sous Python 3.12.

**Problème découvert** : le serveur officiel de téléchargement (CodaLab) a un **certificat TLS expiré depuis le 7 octobre 2026** ; le téléchargement automatique de `wilds` échouera tant que ce n'est pas corrigé. Il existe des copies sur Hugging Face (la plus utilisée : `wltjr1007/Camelyon17-WILDS`, au format parquet, 10,7 Go) dont il faudra vérifier le contenu (mêmes patches, mêmes métadonnées hôpital/lame). Décision en phase 1.

## 3. Règles de rigueur (rappel permanent)

1. **Le test (hôpital 2) ne sert qu'à mesurer, jamais à choisir.** Époque, seuil, hyperparamètre, température de calibration, seuil d'abstention : tout se choisit sur `val` (hôpital 1).
2. **Aucun chiffre inventé** : chaque chiffre du README pointe vers un fichier de `results/` produit par le code.
3. **Graines fixées et passées en paramètre** ; chaque expérience = une commande (`python -m src.train --config configs/X.yaml --seed 1`).
4. **Au moins 3 graines** par configuration, résultat = moyenne ± écart-type, valeurs par graine conservées.
5. README honnête : développé avec l'aide de Claude Code ; limites explicites.

## 4. Architecture du dépôt (cible)

```
README.md, MODEL_CARD.md, LEARNING.md, PLAN.md
requirements.txt            # versions minimales
requirements-lock.txt       # versions exactes ayant produit results/
configs/                    # un YAML par expérience (cnn_none, cnn_hed, cnn_macenko, probe_imagenet, probe_phikon)
src/
  check_env.py              # phase 0 (fait)
  utils.py                  # graines, choix du device, lecture des configs
  data.py                   # chargement WILDS, transformations, HED jitter, Macenko
  features.py               # extraction des caractéristiques figées -> disque
  train.py                  # entraînement CNN et classifieurs linéaires
  evaluate.py               # métriques globales et par hôpital, calibration, risque-couverture
  explain.py                # Grad-CAM, galeries d'erreurs
tests/                      # quelques tests unitaires des métriques (ECE, seuil à spécificité fixée)
notebooks/01_exploration.ipynb
notebooks/kaggle_runner.ipynb   # notebook mince qui appelle src/ sur GPU
results/                    # CSV/JSON/PNG générés, versionnés
artifacts/                  # poids et caractéristiques (non versionnés, .gitignore)
app/app.py                  # Gradio
```

**Données hors du dépôt** : `~/datasets/camelyon17_v1.0/` (modifiable via `DATA_ROOT`). 15 Go n'ont rien à faire dans Git.

## 5. Les phases

### Phase 0 — Environnement et plan (en cours)
Livrables : dépôt, structure, `src/check_env.py`, ce plan, `LEARNING.md`.

### Phase 1 — Données et exploration
- Résoudre la source des données (serveur officiel ou copie vérifiée) ; **tu lances le téléchargement**.
- `notebooks/01_exploration.ipynb` : nombre de patches par split et par hôpital, équilibre des classes, nombre de lames, grilles d'exemples par hôpital, histogrammes de couleur (RGB et espace HED) par hôpital.
- Vérification importante : **des lames sont-elles partagées entre `train` et `id_val` ?** Si oui, `id_val` est optimiste (fuite au niveau lame) — à dire dans le README.
- Figures sauvegardées dans `results/figures/`.

### Phase 2 — CNN de référence + évaluation par hôpital
- `src/utils.py`, `src/data.py`, `src/train.py`, `src/evaluate.py`, `configs/cnn_none.yaml`.
- Mini-banc d'essai chronométré (quelques centaines d'itérations) sur Mac et sur GPU cloud → estimation réelle du coût, puis décision.
- Entraînement selon le protocole WILDS, sélection de l'époque sur `val`.
- Évaluation : précision officielle (évaluateur `wilds`), AUC, sensibilité à spécificité fixée (seuil fixé sur `val`, proposition : spécificité 95 %), **par hôpital et globale**.
- 1 graine pour valider la chaîne, puis 3 graines.

### Phase 3 — Caractéristiques figées + classifieurs linéaires
- Extraction **une seule fois** des caractéristiques de tous les patches pour 2 modèles, sauvegarde sur disque (`artifacts/features/`).
- Modèles : Phikon (`owkin/phikon`, ViT-B/16 auto-supervisé sur de l'histologie TCGA ; **licence non commerciale**, à vérifier sur la fiche) contre un modèle ImageNet.
- Décision à prendre en phase 3 : modèle ImageNet = **ViT-B/16 (DINO)**, même architecture que Phikon, donc seule la nature des données de pré-entraînement change (comparaison propre) ; ou **ResNet-50**, plus classique.
- Piège identifié : une régression logistique est un problème convexe, donc relancer avec 3 graines donne 3 fois le même résultat. Il faudra donner un sens aux graines (tête linéaire entraînée par SGD, ou sous-échantillonnage) — on en discutera.
- Autre limite à documenter : caractéristiques extraites une fois ⇒ pas d'augmentation de données possible.

### Phase 4 — Couleur + graines multiples
- 3 conditions : aucune / augmentation HED (perturbation des canaux hématoxyline-éosine-DAB) / normalisation de Macenko (matrice de référence calculée **sur le train uniquement**).
- 3 conditions × 3 graines = 9 entraînements CNN (les 3 de la phase 2 sont la condition « aucune »).
- Tableau final : moyenne ± écart-type, par hôpital.

### Phase 5 — Calibration, abstention, erreurs, Grad-CAM
- ECE (15 intervalles) + diagrammes de fiabilité ; *temperature scaling* ajusté sur `val`.
- Courbe risque-couverture (confiance = probabilité max) ; seuil d'abstention de l'app choisi sur `val`.
- Galeries des faux négatifs et faux positifs les plus confiants, par hôpital.
- Grad-CAM sur le dernier bloc du DenseNet ; contrôle de bon sens : l'attention est-elle au centre (là où l'étiquette est définie) ?

### Phase 6 — Application Gradio + Hugging Face Spaces
- Dépôt d'image ou exemple par hôpital → probabilité, Grad-CAM, niveau de confiance.
- « Review by a pathologist » sous le seuil choisi en phase 5 ; bandeau « Research tool, not a medical device ».
- CPU gratuit : on déploie le CNN (léger) ; les exemples viennent du jeu CC0.

### Phase 7 — README, fiche du modèle, relecture
- README : une phrase, lien de la démo, résultat clé (tiré de `results/`) … limites, licences, « Reproduce in 3 commands ».
- `MODEL_CARD.md` : usage prévu, données, performances par hôpital, limites, éthique.
- Relecture « comme un recruteur ».

## 6. Calendrier proposé (6 semaines)

| Semaine | Dates | Contenu |
|---|---|---|
| 1 | 17 – 23 oct. | Phases 0 et 1 |
| 2 | 24 – 30 oct. | Phase 2 (+ lancement des graines restantes) |
| 3 | 31 oct. – 6 nov. | Phase 3 ; les entraînements de la phase 4 tournent en parallèle sur GPU |
| 4 | 7 – 13 nov. | Phase 4 (analyse) |
| 5 | 14 – 20 nov. | Phase 5 |
| 6 | 21 – 26 nov. | Phases 6 et 7 |

**Plan B si retard** (dans cet ordre) : abandonner Macenko et garder l'augmentation HED ; réduire les galeries d'erreurs ; garder 3 graines dans tous les cas (sinon les comparaisons ne valent rien).

## 7. Calcul : où tourne quoi

- **Mac M4 (sans ventilateur)** : développement, tests sur petits sous-ensembles, exploration, classifieurs linéaires, évaluation, application. Un entraînement complet y serait long et le processeur ralentirait en chauffant.
- **GPU cloud** : les 9 entraînements CNN et l'extraction des caractéristiques Phikon/ImageNet. Budget de référence : ~2 h par entraînement sur V100 d'après WILDS, donc de l'ordre d'une vingtaine d'heures GPU pour les 9 ; on mesurera nous-mêmes en phase 2 avant de s'engager.

## 8. Décisions à prendre maintenant

1. Gestionnaire d'environnement : **uv** (recommandé) ou **conda**.
2. Plateforme GPU : **Kaggle** (recommandé) ou **Colab**.
3. CNN de départ : **DenseNet-121 sans pré-entraînement, protocole WILDS** (recommandé) ou **ResNet-50 pré-entraîné ImageNet**.

## 9. Risques

| Risque | Parade |
|---|---|
| Serveur de téléchargement officiel cassé | copie vérifiée (comptes par split/hôpital, métadonnées) ; documenté dans le README |
| Forte variance entre graines | 3 graines minimum, valeurs par graine publiées, pas de conclusion sur des écarts < écart-type |
| Quotas GPU gratuits | banc d'essai d'abord ; points de sauvegarde (checkpoints) pour reprendre une session coupée |
| Licence Phikon non commerciale | mentionnée ; l'app publique n'utilise que le CNN |
| Retard | plan B ci-dessus |
