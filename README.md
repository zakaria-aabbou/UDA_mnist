
# Prédiction des données MNIST avec uniquement 100 labels

Dans ce projet, nous proposons des méthodes simples pour entraîner un réseau de neurones semi-supervisée, notre jeu de donnée sera les données MNIST qui est une base de données de chiffres écrits à la main (MNIST). C’est un jeu de données très utilisé en apprentissage
automatique, ce dernier contiens une base d’apprentissage de 60000(x,y) et une base de
test d’environ 10000(x,y) exemples.
L’objectif c’est de faire un réseau de neurones qui va prédire le label qui est sur l’image
avec uniquement 100 labels ça veut dans notre base d’apprentissage on va avoir 59900(x)
et 100(x,y) exemples ça c’est ce qu’on appelle l’apprentissage semi-supervisé qui consiste
à apprendre un model avec à la fois des données qui contient de labels et des données qui
non pas de labels.
Dans ce projet, nous proposons des méthodes simples de formation ded réseau de neurones
de manière semi-supervisée. Ainsi, le but de cette étude est de fournir une information
complète sur l’application de la méthode utilisé dans l’article [1] qui est la méthode RandAug et ses différents application dans le domaine de la synthèse notamment pour faire
de la data augmentation. Notre principale contribution à ce travail est l’application de
cette technique sur un jeu de données MNIST y compris l’efficacité de la méthode pseudo-
étiquette.

Ce référentiel contient une implémentation PyTorch simple et claire des principaux blocs de construction de "[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)" by Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le


## Paramètres

```
--mod:          default='semisup':          Supervised (sup) or semi-supervised training (semisup)
--sup_num:      default=4000:               Number of samples in supervised training set (out of 50K)
--val_num:      default=1000:               Number of samples in validation set (out of 50K)
--rand_seed:    default=89:                 Random seed for dataset shuffle
--sup_aug:      default=['crop', 'hflip']:  Data augmentation for supervised and unsupervised samples (crop, hflip, cutout, randaug)
--unsup_aug:    default=['randaug']:        Data augmentation (Noise) for unsupervised noisy samples (crop, hflip, cutout, randaug)
--bsz_sup:      default=64:                 Batch size for supervised training
--bsz_unsup:    default=448:                Batch size for unsupervised training
--softmax_temp: default=0.4:                Softmax temperature for target distribution (unsup)
--conf_thresh:  default=0.8:                Confidence threshold for target distribution (unsup)
--unsup_loss_w: default=1.0:                Unsupervised loss weight
--max_iter:     default=500000:             Total training iterations
--vis_idx:      default=10:                 Output visualization index
--eval_idx:     default=1000:               Validation index
--out_dir:      default='./output/':        Output directory
```

## Exemples d'exécutions

Pour semi supervised training:
```
python main.py --mod 'semisup' --sup_num 4000 --sup_aug 'crop' 'hflip' --unsup_aug 'randaug' --bsz_sup 64 --bsz_unsup 448
```

Pour supervised training:
```
python main.py --mod 'sup' --sup_num 49000 --sup_aug 'randaug' --bsz_sup 64
```

## Notes

Dans cette implémentation, une partie du code provient des codes sources comme détaillé ci-dessous :
- Wide_ResNet in model.py: https://github.com/wang3702/EnAET/blob/73fd514c74de18c4f7c091012e5cff3a79e1ddbf/Model/Wide_Resnet.py
    - VanillaNet (initially present in guideline code) also works fine. [substitute Wide_ResNet(28, 2, 0.3, 10) with VanillaNet()]
- RandAugment in randAugment.py: https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
    - notre propre implémentation simple de myRandAugment fonctionne également très bien. [substitute RandAugment with myRandAugment]
- EMA in ema.py: https://github.com/chrischute/squad/blob/master/util.py#L174-L220
