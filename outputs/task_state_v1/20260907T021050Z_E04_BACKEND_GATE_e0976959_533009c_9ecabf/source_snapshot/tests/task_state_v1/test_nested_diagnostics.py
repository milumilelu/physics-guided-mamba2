import unittest
import numpy as np
from test_grouping import example
from src.task_state_learning.grouping import components
from src.task_state_learning.diagnostics import crossfit_depth, target_statistics, group_weights, Y


class NestedDiagnosticsTests(unittest.TestCase):
    def data(self):
        frame=components(example()); rng=np.random.default_rng(7)
        frame['D']=rng.normal(size=len(frame))
        for y in Y: frame[y]=rng.normal(size=len(frame))
        return frame

    def test_validation_labels_cannot_change_depth_features(self):
        frame=self.data(); a=frame.iloc[:20]; b=frame.iloc[20:].copy()
        x1,t1,_=crossfit_depth(a,b)
        b['D']=1e9; b[Y]=-1e12
        x2,t2,_=crossfit_depth(a,b)
        np.testing.assert_array_equal(x1,x2);np.testing.assert_array_equal(t1,t2)

    def test_crossfit_prediction_does_not_read_own_response(self):
        frame=self.data(); a=frame.iloc[:20].copy(); b=frame.iloc[20:]
        x,_,trace=crossfit_depth(a,b)
        held=trace[0]['predict_ids']
        a.loc[a.dataset_index.isin(held),'D']=1e10
        changed,_,_=crossfit_depth(a,b)
        idx=a.dataset_index.isin(held).to_numpy()
        np.testing.assert_array_equal(x[idx],changed[idx])

    def test_group_mass_and_shared_ILR_scale(self):
        frame=self.data(); weights=group_weights(frame)
        self.assertAlmostEqual(weights.sum(),1)
        _,scale=target_statistics(frame)
        np.testing.assert_array_equal(scale[1:5],np.repeat(scale[1],4))
        for group in frame.component_id.unique():
            self.assertAlmostEqual(weights[frame.component_id.eq(group)].sum(),1/frame.component_id.nunique())


if __name__=='__main__': unittest.main()
