"""Score unlabelled orders using a frozen evaluation artifact; no refitting."""
import argparse
import json
from pathlib import Path
from risk import FEATURES, score, validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('orders',nargs='?',default='data/new_orders.json')
    parser.add_argument('--artifact',default='example/results.json')
    args=parser.parse_args()
    try:
        rows=json.loads(Path(args.orders).read_text(encoding='utf-8-sig'))
        if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
            raise ValueError('Expected an array of orders')
        # Validation only; artificial label is discarded and never used to score.
        validate([row|{'returned_30d':0} for row in rows])
        artifact=json.loads(Path(args.artifact).read_text(encoding='utf-8-sig'))
        if artifact['model']['features']!=list(FEATURES):
            raise ValueError('Model feature schema mismatch')
        threshold=artifact['results']['logistic']['validation']['threshold']
        print(json.dumps([{'order_id':row['order_id'],'probability':p,'threshold':threshold,'flagged':p>=threshold}
                          for row,p in zip(rows,score(artifact['model'],rows))],indent=2))
    except (ValueError,OSError,TypeError,KeyError) as error:
        parser.exit(1,f'Prediction error: {error}\n')


if __name__=='__main__':
    main()
