import json
import tempfile
import unittest
from pathlib import Path

from research_network.classification import classify
from research_network.config import (ClassificationConfig, Edge, ExperimentConfig, InputConfig,
                                     InputProtocol, ResetType)
from research_network.memory import separability, run_paired
from research_network.metrics import calculate_metrics, detect_period
from research_network.persistence import JsonlStore
from research_network.results import run_experiment
from research_network.simulator import simulate
from research_network.sweep import execute
from research_network.topology import enumerate_input_topologies, enumerate_topologies


def config(**kw):
    base=dict(study_id="test",neuron_count=1,inputs=(InputConfig((0,),1.0,InputProtocol(start_tick=0)),),simulation_ticks=6)
    base.update(kw);return ExperimentConfig(**base)


class DynamicsTests(unittest.TestCase):
    def test_synchronous_one_synapse_per_tick(self):
        c=config(neuron_count=2,recurrent_edges=(Edge(0,1,1),),inputs=(InputConfig((0,)),))
        t=simulate(c).ticks
        self.assertEqual(t[0].spikes,(1,0));self.assertEqual(t[1].spikes,(0,1))

    def test_reset_mechanisms(self):
        expected={ResetType.HARD_RESET:0,ResetType.SUBTRACTIVE_RESET:0.5,ResetType.FIXED_RESIDUAL_RESET:.25,ResetType.PERCENTAGE_RESET:.75}
        for reset,value in expected.items():
            c=config(inputs=(InputConfig((0,),1.5),),reset_type=reset,reset_value=.25,reset_fraction=.5)
            self.assertAlmostEqual(simulate(c).ticks[0].potentials[0],value)

    def test_retention_and_threshold(self):
        c=config(inputs=(InputConfig((0,),.6),),retention=.5,threshold=1.0,simulation_ticks=2)
        t=simulate(c).ticks;self.assertEqual(t[0].spikes,(0,));self.assertAlmostEqual(t[1].potentials[0],.3)
        self.assertEqual(simulate(config(inputs=(InputConfig((0,),1),))).ticks[0].spikes,(1,))

    def test_positive_and_negative_weights(self):
        pos=config(neuron_count=2,recurrent_edges=(Edge(0,1,1),),inputs=(InputConfig((0,),1),))
        neg=config(neuron_count=2,recurrent_edges=(Edge(0,1,-1),),inputs=(InputConfig((0,),1),))
        self.assertEqual(simulate(pos).ticks[1].spikes[1],1);self.assertEqual(simulate(neg).ticks[1].spikes[1],0)


class EnumerationTests(unittest.TestCase):
    def test_topologies_1_2_3(self):
        self.assertEqual(len(enumerate_topologies(1,False)),1)
        self.assertEqual(len(enumerate_topologies(2,False)),4)
        self.assertEqual(len(enumerate_topologies(3,False)),64)
        self.assertEqual(len(enumerate_topologies(1,True)),2)
        self.assertEqual(enumerate_topologies(2)[-1][0],"n2_01-10")

    def test_input_enumeration_excludes_duplicates(self):
        rows=enumerate_input_topologies(3,2)
        self.assertEqual(len(rows),42)
        for _,inputs in rows:self.assertNotEqual(inputs[0].targets,inputs[1].targets)
        self.assertEqual(len(enumerate_input_topologies(1,1)),1)


class IdentityPersistenceTests(unittest.TestCase):
    def test_deterministic_hash_roundtrip(self):
        c=config(recurrent_edges=(Edge(0,0,.1),),retention=.2)
        clone=ExperimentConfig.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(c.config_hash,clone.config_hash)
        self.assertNotEqual(c.config_hash,config(retention=.3).config_hash)

    def test_jsonl_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            store=JsonlStore(Path(d)/"r.jsonl");cs=[config(retention=.1),config(retention=.2)]
            store.append(run_experiment(cs[0]));p=execute(cs,store,workers=1,multiprocessing=False)
            self.assertEqual(p.skipped,1);self.assertEqual(p.completed,1);self.assertEqual(len(list(store.records())),2)
            with store.path.open("a") as f:f.write('{"incomplete":')
            self.assertEqual(len(list(store.records())),2)


class AnalysisTests(unittest.TestCase):
    def test_period_detection(self):
        self.assertEqual(detect_period([0,1,0,1,0,1],0,3,3)[0],2)
        c=config(recurrent_edges=(Edge(0,0,1),),simulation_ticks=10)
        m=calculate_metrics(simulate(c));self.assertEqual(m["full_state_period"],1)
        self.assertEqual(classify(m,c.classification)["primary_regime"],"TONIC")
        dead=calculate_metrics(simulate(config(inputs=(),simulation_ticks=10)))
        self.assertTrue(dead["full_state_period_detected"])
        self.assertFalse(dead["periodic_activity_detected"])
        self.assertEqual(classify(dead,config().classification)["primary_regime"],"DEAD")

    def test_regimes(self):
        cc=ClassificationConfig()
        base={"spike_rate_network":0,"synchrony":0,"total_spikes":0,"activity_lifetime":0,"potential_per_neuron":[{"final":0}],"full_state_period_detected":False,"spike_period_detected":False,"periodic_activity_detected":False,"activity_survived_until_end":False}
        self.assertEqual(classify(base,cc)["primary_regime"],"DEAD")
        self.assertEqual(classify({**base,"potential_per_neuron":[{"final":.2}]},cc)["primary_regime"],"QUIESCENT_WITH_STATE")
        self.assertEqual(classify({**base,"total_spikes":1},cc)["primary_regime"],"DEAD")
        self.assertEqual(classify({**base,"total_spikes":2,"activity_lifetime":1},cc)["primary_regime"],"TRANSIENT")

    def test_paired_separability(self):
        a=simulate(config(inputs=(InputConfig((0,),1),),retention=1,simulation_ticks=4))
        b=simulate(config(inputs=(InputConfig((0,),0),),retention=1,simulation_ticks=4))
        m=separability(a,b)
        self.assertGreater(m["peak_state_separability"],0);self.assertFalse(m["memory_survived_until_end"])
        summary=run_paired(a.config,b.config)
        self.assertNotIn("potential_distance",summary["metrics"])


if __name__ == "__main__": unittest.main()
