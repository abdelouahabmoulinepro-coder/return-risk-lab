import copy
from datetime import timedelta
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from generate_demo import generate
from risk import evaluate,fit,iso,metrics,score,select_threshold,sigmoid,split,validate


class RiskTests(unittest.TestCase):
    def test_stable_sigmoid(self):
        self.assertEqual(sigmoid(0),.5)
        self.assertEqual(sigmoid(-1000),0)
        self.assertEqual(sigmoid(1000),1)

    def test_first_gradient_step_hand_calculated(self):
        base={'delivery_delay_days':0,'fragile':0,'item_price_eur':50}
        rows=[base|{'discount_rate':0,'returned_30d':0},base|{'discount_rate':1,'returned_30d':1}]
        model=fit(rows,epochs=1)
        self.assertEqual(model['weights'][0],0)
        self.assertAlmostEqual(model['weights'][1],.075)
        self.assertEqual(model['scales'][1:],[1,1,1])

    def test_learns_separable_fixture(self):
        rows=[{'discount_rate':i%2,'delivery_delay_days':0,'fragile':0,'item_price_eur':50,'returned_30d':i%2} for i in range(20)]
        model=fit(rows,epochs=100)
        probabilities=score(model,rows)
        self.assertLess(probabilities[0],.2)
        self.assertGreater(probabilities[1],.8)

    def test_metric_confusion_and_ties(self):
        result=metrics([0,1,0,1],[.1,.8,.8,.9],.5)
        self.assertEqual((result['tp'],result['fp'],result['tn'],result['fn']),(2,1,1,0))
        self.assertAlmostEqual(result['auc'],.875)
        self.assertAlmostEqual(result['precision'],2/3)
        self.assertEqual(result['scenario_cost_units'],1)
        self.assertEqual(metrics([0,1],[.4,.4],.5)['auc'],.5)

    def test_undefined_metrics(self):
        result=metrics([0,0],[0,0],.5)
        self.assertIsNone(result['auc'])
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['precision'])

    def test_threshold_ties_prefer_high_cutoff(self):
        self.assertEqual(select_threshold([0,0],[.2,.2]),.95)

    def test_maturity_boundaries_and_disjointness(self):
        parts=split(generate(),'2026-06-01','2026-08-01','2026-10-01')
        self.assertEqual({k:len(v) for k,v in parts.items()},{'train':488,'validation':128,'test':124,'excluded':232})
        for name,boundary in [('train','2026-06-01'),('validation','2026-08-01'),('test','2026-10-01')]:
            self.assertTrue(all(iso(row['delivered_on'])+timedelta(days=30)<=iso(boundary) for row in parts[name]))
        self.assertEqual(len({r['order_id'] for part in parts.values() for r in part}),972)

    def test_future_labels_cannot_change_fit_or_threshold(self):
        records=generate()
        original=evaluate(records)
        changed=copy.deepcopy(records)
        for row in changed:
            if row['delivered_on']>='2026-08-01':
                row['returned_30d']=1-row['returned_30d']
        revised=evaluate(changed)
        self.assertEqual(original['model'],revised['model'])
        for name in original['results']:
            self.assertEqual(original['results'][name]['validation'],revised['results'][name]['validation'])
        self.assertEqual([r['probability'] for r in original['test_predictions']],[r['probability'] for r in revised['test_predictions']])

    def test_standardization_uses_training_only(self):
        parts=split(generate(),'2026-06-01','2026-08-01','2026-10-01')
        model=fit(parts['train'],epochs=1)
        expected=sum(r['item_price_eur'] for r in parts['train'])/len(parts['train'])
        self.assertAlmostEqual(model['means'][3],expected)
        previous=copy.deepcopy(model)
        score(model,[parts['test'][0]|{'item_price_eur':100000}])
        self.assertEqual(model,previous)

    def test_invalid_inputs(self):
        for changes in [{'fragile':True},{'returned_30d':2},{'discount_rate':float('nan')},{'delivered_on':'2026-02-30'}]:
            rows=generate()
            rows[0].update(changes)
            with self.assertRaises(ValueError): validate(rows)
        rows=generate()
        rows.append(rows[0].copy())
        with self.assertRaises(ValueError): validate(rows)

    def test_model_json_roundtrip(self):
        parts=split(generate(),'2026-06-01','2026-08-01','2026-10-01')
        model=fit(parts['train'],epochs=10)
        self.assertEqual(score(model,parts['test']),score(json.loads(json.dumps(model)),parts['test']))

    def test_cli_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'risk.py','--output',folder],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((Path(folder)/'results.json').read_text())
            self.assertEqual(report['results']['logistic']['test']['n'],124)
            self.assertTrue((Path(folder)/'report.md').exists())


if __name__=='__main__':
    unittest.main()
