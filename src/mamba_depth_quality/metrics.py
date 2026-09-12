import numpy as np

def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def component_balanced_mae(y,p,components):
    e=np.abs(np.asarray(y)-np.asarray(p)); return float(np.mean([e[np.asarray(components)==c].mean() for c in np.unique(components)]))
def summarize(y,p,components=None):
    out={"MAE":mae(y,p),"RMSE":rmse(y,p)}
    if components is not None: out["component_balanced_MAE"]=component_balanced_mae(y,p,components)
    return out
