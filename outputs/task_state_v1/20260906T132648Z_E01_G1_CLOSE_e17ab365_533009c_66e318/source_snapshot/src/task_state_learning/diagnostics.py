"""E02 grouped, nested static diagnostics; no hidden-state or physics training."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from .grouping import assign_folds, require

U = ["pulse_duration_fs", "frequency_kHz", "velocity_mm_s", "hatch_spacing_um", "pass_count"]
Y = ["A_med", "ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4", "A2_8_16", "entropy_8_16"]
VARIANTS = ["M_D", "M_Dh", "M_U", "M_UD", "M_UDhat", "C1", "C2", "C3", "C4"]


def group_weights(frame):
    counts = frame.groupby("component_id").component_id.transform("size").to_numpy()
    return 1.0 / (frame.component_id.nunique() * counts)


def target_statistics(frame):
    y = frame[Y].to_numpy(float); w = group_weights(frame)
    mean = w @ y
    variance = np.maximum(w @ ((y-mean)**2), 1e-12)
    variance[1:5] = np.mean(variance[1:5])
    return mean, np.sqrt(variance)


def per_row_risk(y, pred, scale):
    error = ((np.asarray(y)-pred)/scale)**2
    return (error[:,0] + error[:,1:5].mean(axis=1) + error[:,5:7].mean(axis=1))/3


@dataclass
class Regressor:
    family: str
    alpha: float

    def fit(self, x, y, weights):
        x=np.asarray(x,float); y=np.asarray(y,float)
        self.mean=weights @ x
        self.scale=np.sqrt(np.maximum(weights @ ((x-self.mean)**2),1e-12))
        xs=(x-self.mean)/self.scale
        self.spline=None
        if self.family=='GAM':
            self.spline=SplineTransformer(n_knots=4,degree=2,include_bias=False,knots='uniform',extrapolation='linear')
            xs=self.spline.fit_transform(xs)
        self.model=Ridge(alpha=self.alpha)
        # Fixed effective mass equal to sample count, with component-balanced contributions.
        self.model.fit(xs,y,sample_weight=weights*len(weights))
        return self

    def predict(self,x):
        xs=(np.asarray(x,float)-self.mean)/self.scale
        if self.spline is not None: xs=self.spline.transform(xs)
        return self.model.predict(xs)


def raw_features(frame, variant, dhat=None):
    u=frame[U].to_numpy(float)
    require((u>0).all(),'Nonpositive process input')
    logu=np.log(u)
    if variant=='M_D': return frame[['D']].to_numpy(float)
    if variant=='M_Dh': return np.column_stack([frame.D,logu[:,3]])
    if variant=='M_U': return logu
    if variant=='M_UD': return np.column_stack([logu,frame.D])
    if variant=='M_UDhat':
        require(dhat is not None and len(dhat)==len(frame),'Missing cross-fitted depth')
        return np.column_stack([logu,dhat])
    tau,f,v,h,n=u.T
    qa=1000*5.3333*n/(v*h); dx=v/f; ep=1000*5.3333/f
    columns={'C1':[qa],'C2':[qa,h],'C3':[qa,h,dx],'C4':[qa,h,dx,ep]}
    return np.log(np.column_stack(columns[variant]))


def crossfit_depth(train, evaluation, alpha=1.0, return_model=False):
    """Fit all depth features without evaluation responses; depth alpha is fixed."""
    fold=assign_folds(train,3).to_numpy()
    result=np.full(len(train),np.nan); trace=[]
    for f in range(3):
        a=train.iloc[np.flatnonzero(fold!=f)]; b=train.iloc[np.flatnonzero(fold==f)]
        require(not set(a.component_id)&set(b.component_id),'Depth crossfit leakage')
        model=Regressor('Ridge',alpha).fit(raw_features(a,'M_U'),a.D.to_numpy(),group_weights(a))
        result[fold==f]=model.predict(raw_features(b,'M_U'))
        trace.append({'fit_ids':list(map(int,a.dataset_index)),'predict_ids':list(map(int,b.dataset_index))})
    model=Regressor('Ridge',alpha).fit(raw_features(train,'M_U'),train.D.to_numpy(),group_weights(train))
    predicted=model.predict(raw_features(evaluation,'M_U'))
    trace.append({'fit_ids':list(map(int,train.dataset_index)),'predict_ids':list(map(int,evaluation.dataset_index))})
    require(np.isfinite(result).all(),'Incomplete cross-fitted depth')
    return (result,predicted,trace,model) if return_model else (result,predicted,trace)


def fit_predict(train, evaluation, variant, family, alpha, depth_features=None, return_model=False):
    if variant=='M_UDhat':
        require(depth_features is not None,'Dhat must be cross-fitted')
        a,b=depth_features
    else: a=b=None
    model=Regressor(family,alpha).fit(raw_features(train,variant,a),train[Y].to_numpy(float),group_weights(train))
    pred=model.predict(raw_features(evaluation,variant,b))
    require(np.isfinite(pred).all(),'Nonfinite static prediction')
    return (pred,model) if return_model else pred
