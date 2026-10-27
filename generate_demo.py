"""Fictional delivered orders; no customer information or real return records."""
from datetime import date,timedelta
import json
from pathlib import Path
import random
from risk import sigmoid


def generate():
    rng=random.Random(20261027)
    rows=[]
    for day in range(243):
        for _ in range(4):
            discount=round(rng.uniform(0,.6),2)
            delay=rng.choice([0,0,0,1,2,3,5,8])
            fragile=int(rng.random()<.3)
            price=rng.randint(15,300)
            probability=sigmoid(-3.5+2.5*discount+.25*delay+.8*fragile+.003*price)
            rows.append({'order_id':f'O{len(rows)+1:04}','delivered_on':str(date(2026,1,1)+timedelta(days=day)),
                         'discount_rate':discount,'delivery_delay_days':delay,'fragile':fragile,
                         'item_price_eur':price,'returned_30d':int(rng.random()<probability)})
    return rows


if __name__=='__main__':
    path=Path('data/orders.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(generate(),indent=2)+'\n',encoding='utf-8')
    print('Created 972 fictional orders')
