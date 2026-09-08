"""End-to-end counterexamples for the audited training/evaluation failures."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.task_state_learning.losses import decision_gap_loss,PairSampler
from src.task_state_learning.models import B1ClosureModel
from src.task_state_learning.training import fit,predict
from src.task_state_learning.benchmarks import b1_reference


def runner(name):
    path=Path(__file__).resolve().parents[2]/'experiments/task_state_v1'/name
    spec=importlib.util.spec_from_file_location('repair_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RepairChainTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def data(self):
        g=torch.Generator().manual_seed(123)
        return {'control':torch.rand(8,4,2,generator=g)*1.6-.8,
                'physical_dt':torch.full((8,4),.02),'mask':torch.ones(8,4,dtype=torch.bool),
                'terminal_x':torch.rand(8,2,generator=g)*.2-.1}

    def test_gap_exact_prediction_is_zero_and_reversal_penalized(self):
        y=torch.tensor([[0.,0.],[1.,1.]],requires_grad=True)
        anchor=torch.zeros(1,2);pairs=torch.tensor([[0,1]])
        self.assertEqual(float(decision_gap_loss(y,y.detach(),anchor,torch.ones(2),pairs).detach()),0.)
        self.assertGreater(float(decision_gap_loss(y.flip(0),y.detach(),anchor,torch.ones(2),pairs).detach()),0.)

    def test_resume_matches_uninterrupted_joint_training_and_reload(self):
        data=self.data()
        dp={'anchors':data['terminal_x'][:2],'sampler':PairSampler(8,4,17)}
        def make():
            torch.manual_seed(17);return B1ClosureModel('GRU')
        with tempfile.TemporaryDirectory() as folder:
            uninterrupted=fit(make(),data,data,lr=1e-3,seed=17,fixed_epochs=2,dp=dp,
                               batch=2,accum=2,checkpoint_dir=Path(folder)/'full')
            def interrupt(record):raise RuntimeError('simulated interruption')
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                fit(make(),data,data,lr=1e-3,seed=17,fixed_epochs=2,dp=dp,batch=2,accum=2,
                    checkpoint_dir=Path(folder)/'resume',epoch_callback=interrupt)
            resumed=fit(make(),data,data,lr=1e-3,seed=17,fixed_epochs=2,dp=dp,batch=2,accum=2,
                        checkpoint_dir=Path(folder)/'resume')
            for key,value in uninterrupted['model'].state_dict().items():
                torch.testing.assert_close(value,resumed['model'].state_dict()[key],rtol=0,atol=0)
            loaded=make();loaded.load_state_dict(torch.load(Path(folder)/'resume/selected_state.pt',weights_only=True))
            torch.testing.assert_close(predict(loaded,data),predict(resumed['model'],data),rtol=0,atol=0)

    def test_sealed_design_can_select_either_candidate(self):
        module=runner('04_b1_terminal.py')
        module.predict=lambda model,data:data['terminal_x']
        data={'terminal_x':torch.tensor([[0.,0.],[1.,1.]])}
        with tempfile.TemporaryDirectory() as folder:
            result,_=module.sealed_design(None,data,torch.zeros(1,2),torch.ones(2),None,Path(folder),'test',17)
            self.assertEqual(len(result),256)
            self.assertTrue((result.regret==0).all())

    def test_zero_control_coupled_continuation_is_not_free_decay(self):
        initial=np.array([[0.,0.,1.,1.]])
        result=b1_reference(np.zeros((1,1,2)),np.full((1,1),.1),initial=initial)
        self.assertGreater(np.linalg.norm(result[0,:2]),.001)

    def test_closure_regularizer_has_gradient(self):
        model=B1ClosureModel('GRU')
        torch.nn.init.constant_(model.readout[-1].bias,.1)
        data=self.data()
        _,penalty=model(data['control'],data['physical_dt'],data['mask'],return_closure=True)
        penalty.backward()
        self.assertGreater(float(model.readout[-1].bias.grad.abs().sum()),0.)

    def test_calibration_rejects_bounds_and_uses_registered_start_count(self):
        import yaml
        module=runner('03_physics_calibration.py')
        cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'config/task_state_v1/physics_calibration.yaml').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError,'out of bounds'):
            module.physics_params([10.,-20.,2.,2.],'P00',.874e-6,cfg)
        self.assertEqual(len(module.starts('P11',np.random.default_rng(17),cfg)),cfg['calibration']['n_starts'])

    def test_calibration_component_duplication_keeps_equal_group_weight(self):
        import pandas as pd,yaml
        module=runner('03_physics_calibration.py')
        cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'config/task_state_v1/physics_calibration.yaml').read_text(encoding='utf-8'))
        frame=pd.DataFrame({'dataset_index':[0,1],'component_id':['a','b'],'D':[1.,2.],'A_med':[.1,.3]})
        repeated=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
        with patch.object(module,'simulate_row',return_value=(.8,.2,{})):
            a=module.Objective(frame,'P00',.874e-6,cfg)([10.,-20.,0.,0.])
            b=module.Objective(repeated,'P00',.874e-6,cfg)([10.,-20.,0.,0.])
        self.assertAlmostEqual(a,b,places=12)

    def test_calibration_plan_covers_all_inner_folds_without_outer_test(self):
        import pandas as pd,yaml
        module=runner('03_physics_calibration.py')
        root=Path(__file__).resolve().parents[2]
        cfg=yaml.safe_load((root/'config/task_state_v1/physics_calibration.yaml').read_text(encoding='utf-8'))
        folder=root/'outputs/task_state_v1'/cfg['split_run']
        manifest=pd.read_csv(folder/'split_manifest.csv');inner=pd.read_csv(folder/'inner_split_manifest.csv')
        jobs=module.build_jobs(manifest,manifest,inner,cfg)
        self.assertEqual(len(jobs),240)
        self.assertEqual(len({name for name,_,_ in jobs}),240)
        for _,level,payload in jobs:
            fold,_,_,_,train,validation,subset,_=payload
            outer_test=set(manifest.loc[manifest.outer_fold==fold,'dataset_index'])
            self.assertFalse(set(train)&outer_test)
            self.assertFalse(set(train)&set(validation))
            self.assertTrue(set(subset)<=set(train))
            if level!=-1:self.assertFalse(set(validation)&outer_test)

    def test_static_features_do_not_change_with_padding(self):
        from src.task_state_learning.static_baseline import StaticMLP
        model=StaticMLP();g=torch.Generator().manual_seed(5)
        u=torch.rand(2,32,2,generator=g)
        dt=torch.full((2,32),.02);mask=torch.ones(2,32,dtype=torch.bool)
        a=model.features(u,dt,mask)
        b=model.features(torch.cat((u,torch.zeros(2,96,2)),1),
                         torch.cat((dt,torch.zeros(2,96)),1),
                         torch.cat((mask,torch.zeros(2,96,dtype=torch.bool)),1))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_duplicating_pairs_does_not_increase_bootstrap_sample_size(self):
        from src.task_state_learning.design import family_pair_interval
        links=np.array([(i,j) for i in range(5) for j in range(i+1,5)])
        values=np.arange(len(links),dtype=float)
        a=family_pair_interval(values,links,bootstrap=1000)
        b=family_pair_interval(np.tile(values,2),np.tile(links,(2,1)),bootstrap=1000)
        self.assertEqual(a['n_families'],5)
        self.assertEqual(a['low'],b['low'])
        self.assertEqual(a['high'],b['high'])


if __name__=='__main__':unittest.main()
