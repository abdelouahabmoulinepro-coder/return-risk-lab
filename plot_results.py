"""Optional visual audit of the frozen test predictions."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',nargs='?',default='example/results.json')
    parser.add_argument('--output',default='example/return-risk-audit.png')
    args=parser.parse_args()
    data=json.loads(Path(args.input).read_text(encoding='utf-8'))
    result=data['results']['logistic']['test']
    predictions=data['test_predictions']
    fig,(left,right)=plt.subplots(1,2,figsize=(11,5.4),layout='constrained',facecolor='#f8fafc')
    bins=[i/10 for i in range(11)]
    for target,color,label in [(0,'#64748b','Sans retour'),(1,'#f97316','Avec retour')]:
        left.hist([row['probability'] for row in predictions if row['actual']==target],bins=bins,color=color,alpha=.65,label=label)
    left.axvline(result['threshold'],color='#0f172a',linestyle='--',label=f"Seuil {result['threshold']:.2f}")
    left.set_title('Distribution des scores sur le test',loc='left',pad=12)
    left.set_xlabel('Probabilité estimée de retour')
    left.set_ylabel('Nombre de commandes fictives')
    left.legend(fontsize=9)
    matrix=[[result['tn'],result['fp']],[result['fn'],result['tp']]]
    right.imshow(matrix,cmap='Blues',vmin=0,vmax=max(max(row) for row in matrix))
    for i in range(2):
        for j in range(2):
            right.text(j,i,str(matrix[i][j]),ha='center',va='center',fontsize=30,color='white' if matrix[i][j]>40 else '#0f172a')
    right.set_xticks([0,1],['Non signalé','Signalé'])
    right.set_yticks([0,1],['Sans retour','Avec retour'])
    right.set_xlabel('Décision au seuil fixé sur validation')
    right.set_ylabel('Résultat observé à 30 jours')
    right.set_title(f"124 commandes · AUC {result['auc']:.3f}",loc='left',pad=12)
    fig.suptitle('RETURN RISK LAB\nPrioriser sans masquer les retours manqués',fontsize=18,weight='bold')
    fig.supxlabel('Données synthétiques · 488 apprentissage / 128 validation / 124 test · 232 exclus pour maturité des labels',fontsize=9,color='#475569')
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    plt.close(fig)
    print(output)


if __name__=='__main__':
    main()
