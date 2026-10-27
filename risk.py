"""Educational return-risk classifier with delayed-label chronological evaluation."""
import argparse
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean

FEATURES = ('discount_rate', 'delivery_delay_days', 'fragile', 'item_price_eur')


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def iso(value):
    if not isinstance(value,str):
        raise ValueError('Dates must be strings')
    day=date.fromisoformat(value)
    if day.isoformat()!=value:
        raise ValueError('Dates must use YYYY-MM-DD')
    return day


def validate(rows):
    if not isinstance(rows,list) or not rows:
        raise ValueError('Expected a nonempty list of orders')
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('order_id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',row['order_id']):
            raise ValueError('Each order requires a short ASCII order_id')
        if row['order_id'] in seen:
            raise ValueError('Duplicate order_id')
        seen.add(row['order_id'])
        iso(row.get('delivered_on'))
        for key,high in [('discount_rate',1),('delivery_delay_days',90),('fragile',1),('item_price_eur',100000)]:
            value=row.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=high:
                raise ValueError(f'Invalid feature {key}')
        if type(row['fragile']) is not int or row['fragile'] not in (0,1):
            raise ValueError('fragile must be integer 0 or 1')
        if type(row.get('returned_30d')) is not int or row['returned_30d'] not in (0,1):
            raise ValueError('returned_30d must be integer 0 or 1')


def split(rows,validation_start,test_start,as_of):
    validate(rows)
    validation_start,test_start,as_of=map(iso,(validation_start,test_start,as_of))
    if not validation_start<test_start<as_of:
        raise ValueError('Require validation_start < test_start < as_of')
    parts={'train':[],'validation':[],'test':[],'excluded':[]}
    for row in sorted(rows,key=lambda r:(r['delivered_on'],r['order_id'])):
        delivered=iso(row['delivered_on'])
        mature=delivered+timedelta(days=30)
        if delivered<validation_start and mature<=validation_start:
            target='train'
        elif validation_start<=delivered<test_start and mature<=test_start:
            target='validation'
        elif test_start<=delivered and mature<=as_of:
            target='test'
        else:
            target='excluded'
        parts[target].append(row)
    for name in ('train','validation','test'):
        if not parts[name]:
            raise ValueError(f'No mature orders in {name}')
    if {r['returned_30d'] for r in parts['train']}!={0,1}:
        raise ValueError('Training requires both classes')
    return parts


def fit(rows,epochs=500,learning_rate=.15,l2=.01):
    """Fixed batch-gradient logistic regression; standardization fits training only."""
    if not rows or epochs<1 or learning_rate<=0 or l2<0:
        raise ValueError('Invalid fitting parameters')
    means=[mean(row[key] for row in rows) for key in FEATURES]
    scales=[math.sqrt(mean((row[key]-mu)**2 for row in rows)) or 1 for key,mu in zip(FEATURES,means)]
    x=[[1]+[(row[key]-mu)/scale for key,mu,scale in zip(FEATURES,means,scales)] for row in rows]
    y=[row['returned_30d'] for row in rows]
    weights=[0.0]*(len(FEATURES)+1)
    for _ in range(epochs):
        grad=[0.0]*len(weights)
        for vector,target in zip(x,y):
            residual=sigmoid(sum(w*v for w,v in zip(weights,vector)))-target
            for j,value in enumerate(vector):
                grad[j]+=residual*value/len(rows)
        for j in range(len(weights)):
            weights[j]-=learning_rate*(grad[j]+(l2*weights[j] if j else 0))
    return {'features':list(FEATURES),'means':means,'scales':scales,'weights':weights,
            'epochs':epochs,'learning_rate':learning_rate,'l2':l2}


def score(model,rows):
    return [sigmoid(model['weights'][0]+sum(w*(row[key]-mu)/scale for w,key,mu,scale in
            zip(model['weights'][1:],FEATURES,model['means'],model['scales']))) for row in rows]


def metrics(labels,probabilities,threshold):
    if not labels or len(labels)!=len(probabilities):
        raise ValueError('Nonempty, equal-length labels and probabilities required')
    if any(type(y) is not int or y not in (0,1) for y in labels) or any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
        raise ValueError('Invalid labels or probabilities')
    if not 0<=threshold<=1:
        raise ValueError('Threshold outside [0,1]')
    tp=fp=tn=fn=0
    for y,p in zip(labels,probabilities):
        if p>=threshold:
            if y: tp+=1
            else: fp+=1
        elif y: fn+=1
        else: tn+=1
    positive=[p for y,p in zip(labels,probabilities) if y]
    negative=[p for y,p in zip(labels,probabilities) if not y]
    auc=mean((a>b)+.5*(a==b) for a in positive for b in negative) if positive and negative else None
    clipped=[min(1-1e-15,max(1e-15,p)) for p in probabilities]
    return {'n':len(labels),'positives':sum(labels),'threshold':threshold,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'flagged':tp+fp,'auc':auc,'brier':mean((p-y)**2 for y,p in zip(labels,probabilities)),
            'log_loss':-mean(y*math.log(p)+(1-y)*math.log(1-p) for y,p in zip(labels,clipped)),
            'scenario_cost_units':fp+5*fn}


def select_threshold(labels,probabilities):
    # Costs fixed before test: false positive = 1, missed return = 5 arbitrary units.
    candidates=[i/20 for i in range(1,20)]
    return min(candidates,key=lambda t:(metrics(labels,probabilities,t)['scenario_cost_units'],-t))


def evaluate(rows,validation_start='2026-06-01',test_start='2026-08-01',as_of='2026-10-01'):
    parts=split(rows,validation_start,test_start,as_of)
    model=fit(parts['train'])
    prevalence=mean(row['returned_30d'] for row in parts['train'])
    outputs={}
    for name in ('constant_prevalence','logistic'):
        valp=[prevalence]*len(parts['validation']) if name=='constant_prevalence' else score(model,parts['validation'])
        testp=[prevalence]*len(parts['test']) if name=='constant_prevalence' else score(model,parts['test'])
        threshold=select_threshold([r['returned_30d'] for r in parts['validation']],valp)
        outputs[name]={'validation':metrics([r['returned_30d'] for r in parts['validation']],valp,threshold),
                       'test':metrics([r['returned_30d'] for r in parts['test']],testp,threshold)}
    predictions=[{'order_id':r['order_id'],'delivered_on':r['delivered_on'],
                  'label_available_on':(iso(r['delivered_on'])+timedelta(days=30)).isoformat(),
                  'actual':r['returned_30d'],'probability':p,
                  'flagged':p>=outputs['logistic']['test']['threshold']} for r,p in zip(parts['test'],score(model,parts['test']))]
    return {'input_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
            'dates':{'validation_start':validation_start,'test_start':test_start,'as_of':as_of},
            'split_counts':{name:len(part) for name,part in parts.items()},
            'split_ids':{name:[r['order_id'] for r in part] for name,part in parts.items()},
            'model':model,'baseline_prevalence':prevalence,'results':outputs,'test_predictions':predictions}


def markdown(report):
    lines=['# Return Risk Lab', '', '> Expérience sur données entièrement synthétiques. Aucun gain opérationnel réel démontré.', '',
           f"Découpage : {report['split_counts']}", '', '| Méthode | AUC | Brier | Seuil | Signalés | Précision | Rappel | Coût hypothétique |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:'—' if value is None else f'{value:.3f}'
    for name,result in report['results'].items():
        m=result['test']
        lines.append(f"| {name} | {fmt(m['auc'])} | {m['brier']:.3f} | {m['threshold']:.2f} | {m['flagged']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {m['scenario_cost_units']} |")
    lines+=['', 'Les seuils minimisent FP + 5 × FN sur validation uniquement. Ce coût en unités arbitraires ne mesure pas une économie financière.', '',
            'Le modèle est ajusté uniquement sur apprentissage, puis reste figé. Les commandes dont le label à 30 jours ne pouvait être connu à la frontière du bloc sont exclues.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',nargs='?',default='data/orders.json')
    parser.add_argument('--validation-start',default='2026-06-01')
    parser.add_argument('--test-start',default='2026-08-01')
    parser.add_argument('--as-of',default='2026-10-01')
    parser.add_argument('--output',default='build')
    args=parser.parse_args()
    try:
        source,output=Path(args.input).resolve(),Path(args.output).resolve()
        if source in [output/'results.json',output/'report.md']:
            raise ValueError('Output would overwrite input')
        result=evaluate(json.loads(source.read_text(encoding='utf-8-sig')),args.validation_start,args.test_start,args.as_of)
        output.mkdir(parents=True,exist_ok=True)
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        (output/'report.md').write_text(markdown(result),encoding='utf-8')
        print(json.dumps({'split_counts':result['split_counts'],'results':result['results']},indent=2))
    except (ValueError,OSError,TypeError) as error:
        parser.exit(1,f'Data/evaluation error: {error}\n')


if __name__=='__main__':
    main()
