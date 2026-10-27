# Return Risk Lab

> Expérience sur données entièrement synthétiques. Aucun gain opérationnel réel démontré.

Découpage : {'train': 488, 'validation': 128, 'test': 124, 'excluded': 232}

| Méthode | AUC | Brier | Seuil | Signalés | Précision | Rappel | Coût hypothétique |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant_prevalence | 0.500 | 0.184 | 0.20 | 124 | 0.242 | 1.000 | 94 |
| logistic | 0.711 | 0.161 | 0.30 | 36 | 0.444 | 0.533 | 90 |

Les seuils minimisent FP + 5 × FN sur validation uniquement. Ce coût en unités arbitraires ne mesure pas une économie financière.

Le modèle est ajusté uniquement sur apprentissage, puis reste figé. Les commandes dont le label à 30 jours ne pouvait être connu à la frontière du bloc sont exclues.
