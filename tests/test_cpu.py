"""Small mathematical/contract checks. No model, experiment or GPU is loaded."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wmmd_bench.methods.ctcc.evaluate import aggregate as ctcc
from wmmd_bench.methods.evertracer.evaluate import aggregate as evertracer
from wmmd_bench.methods.awm.evaluate import aggregate_layers

class DetectorContracts(unittest.TestCase):
    def test_ctcc_exact_rule_and_errors(self):
        rows=[{'category':g,'raw_generation':' IAMALIVE\n' if i==0 else 'IAMALIVE extra'} for g,n in [('trigger',95),('suppression',100),('normal',105)] for i in range(n)]
        out=ctcc(rows)
        self.assertEqual(out['categories']['trigger']['activations'],1)
        self.assertEqual(out['combined_negatives']['false_activation_rate'],2/205)
        rows[0]['generation_error']='inference failed'
        with self.assertRaises(ValueError):ctcc(rows)
    def test_evertracer_ties_reject_all(self):
        rows=[{'subset':g,'calibrated_score':0.} for g in ['dtr','dunseen'] for _ in range(100)]
        out=evertracer(rows)
        self.assertEqual(out['member_oriented_auc'],.5)
        self.assertEqual(out['member_oriented_threshold'],'Infinity')
        self.assertEqual(out['member_oriented_tpr_at_fpr_limit'],0)
        with self.assertRaises(ValueError):evertracer(rows[:-1])
    def test_awm_alignment_cannot_duplicate_layers(self):
        row={'reference_layer':0,'candidate_layer':0,'Wq_weights':.8,'Wk_weights':1.}
        self.assertEqual(aggregate_layers({'per_layer':[row]}),.9)
        with self.assertRaises(ValueError):aggregate_layers({'per_layer':[row,row]})
    def test_response_objective_forward_and_gradient(self):
        from wmmd_bench.training.objective import objective_test
        self.assertEqual(objective_test()['status'],'PASS')

if __name__=='__main__':unittest.main()
