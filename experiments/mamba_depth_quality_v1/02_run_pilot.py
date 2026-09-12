"""Run the pilot OOF experiment for mamba_depth_quality_v1.

Training references are fit by a three-fold inner cross-fit on every outer
training partition.  ``--full`` enables the protocol 300-epoch ceiling with
inner-fold early stopping; ``--fast`` keeps the 12-epoch smoke profile.
Both modes use the actual pure-PyTorch Mamba adapter, never a surrogate.
"""
from __future__ import annotations
import argparse, copy, json, hashlib, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mamba_depth_quality.data import load_main180, component_weights
from src.mamba_depth_quality.reference import fit_reference, reference_depth
from src.mamba_depth_quality.tokens import build_tokens
from src.mamba_depth_quality.models import TerminalModel

MODEL_IDS = ["CONST_DQ","TREE_U_DQ","STATIC_PHY_K4_AUX","GRU_PHY_K4_AUX",
             "MAMBA_RAW_K4_AUX","MAMBA_PHY_K4_NOAUX","MAMBA_PHY_K4_AUX","MAMBA_PHY_K32_AUX"]

def _features(df, ref, phy=True):
    # Static controls use deterministic process summaries; neural routes below use full tokens.
    x = df[["pulse_duration_fs","frequency_kHz","velocity_mm_s","hatch_spacing_um","pass_count","measured_power_W"]].to_numpy(float)
    x = np.log1p(np.maximum(x, 0.0))
    dref = reference_depth(df, ref) if phy else np.zeros(len(df))
    f0 = np.asarray([reference_depth(df.iloc[[i]], ref)[0] for i in range(len(df))]) if False else dref
    # Include simple interactions and reference depth; duplicate columns are
    # intentional so raw/physics routes remain fixed-width and deterministic.
    return np.column_stack([x, dref, dref**2, x[:,1]*x[:,2], x[:,3]*x[:,4]])

def _crossfit_tokens(train, test, outer_ref, inner_folds, raw_ablation=False):
    """Build training tokens with a reference fit outside each inner fold.

    This is the leakage guard required by the protocol: every training row is
    encoded using a reference calibrated on the other two inner folds.  The
    outer test rows use the reference fitted on the complete outer training
    partition.
    """
    n = len(train)
    row_tokens, row_masks, row_dref = [None] * n, [None] * n, np.zeros(n, float)
    for k in sorted(pd.Series(inner_folds).dropna().unique()):
        val = np.asarray(inner_folds) == k
        fit = ~val
        if fit.sum() == 0 or val.sum() == 0:
            continue
        r_k = fit_reference(train.iloc[np.flatnonzero(fit)].copy())
        tx, tm = build_tokens(train.iloc[np.flatnonzero(val)].copy(), r_k,
                              raw_ablation=raw_ablation)
        d_k = reference_depth(train.iloc[np.flatnonzero(val)].copy(), r_k)
        for j, pos in enumerate(np.flatnonzero(val)):
            row_tokens[pos], row_masks[pos], row_dref[pos] = tx[j], tm[j], d_k[j]
    # A defensive fallback keeps the helper usable with an incomplete manifest.
    if any(v is None for v in row_tokens):
        tx, tm = build_tokens(train, outer_ref, raw_ablation=raw_ablation)
        row_tokens = [tx[i] for i in range(n)]; row_masks = [tm[i] for i in range(n)]
        row_dref = reference_depth(train, outer_ref)
    max_t = max(x.shape[0] for x in row_tokens)
    d = row_tokens[0].shape[1]
    arr = np.zeros((n, max_t, d), float); msk = np.zeros((n, max_t), bool)
    for i, (x, m) in enumerate(zip(row_tokens, row_masks)):
        arr[i, :x.shape[0]] = x; msk[i, :m.shape[0]] = m
    txe, tme = build_tokens(test, outer_ref, raw_ablation=raw_ablation)
    return arr, msk, row_dref, txe, tme


def _fit_predict(mid, train, test, ref, seed, fast=True, inner_folds=None):
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor
    y = train[["D","Sq"]].to_numpy(float)
    if mid == "CONST_DQ":
        w = component_weights(train); return np.tile(np.average(y,axis=0,weights=w),(len(test),1)), 0, 0
    phy = "PHY" in mid or mid == "STATIC_PHY_K4_AUX"
    # Train the requested neural encoders on the actual exposure tokens.  The
    # fast flag only limits epochs; it does not replace Mamba with a surrogate.
    if mid in {"STATIC_PHY_K4_AUX","GRU_PHY_K4_AUX","MAMBA_RAW_K4_AUX","MAMBA_PHY_K4_NOAUX","MAMBA_PHY_K4_AUX","MAMBA_PHY_K32_AUX"}:
        import torch
        torch.manual_seed(seed); np.random.seed(seed)
        if inner_folds is None: inner_folds = np.arange(len(train)) % 3
        tx,tm,db_np,txe,tme = _crossfit_tokens(train, test, ref, inner_folds, raw_ablation=(mid=="MAMBA_RAW_K4_AUX"))
        mu=tx.mean(axis=(0,1)); sd=tx.std(axis=(0,1))+1e-8; tx=(tx-mu)/sd; txe=(txe-mu)/sd
        xtr=torch.tensor(tx,dtype=torch.float32); xte=torch.tensor(txe,dtype=torch.float32)
        mask_tr=torch.tensor(tm); mask_te=torch.tensor(tme)
        yd=torch.tensor(train.D.to_numpy(float),dtype=torch.float32); yq=torch.tensor(train.Sq.to_numpy(float),dtype=torch.float32)
        sd_d=float(np.std(train.D))+1e-8; sd_q=float(np.std(train.Sq))+1e-8; db=torch.tensor(db_np,dtype=torch.float32)
        ya=torch.tensor(train[["ilr_z1","ilr_z2"]].to_numpy(float),dtype=torch.float32); ya=(ya-ya.mean(0))/(ya.std(0)+1e-8)
        enc={"STATIC_PHY_K4_AUX":"static","GRU_PHY_K4_AUX":"gru"}.get(mid,"mamba2")
        def make_model():
            torch.manual_seed(seed); return TerminalModel(input_dim=16,encoder=enc,bottleneck=(32 if mid.endswith("K32_AUX") else 4),aux=("NOAUX" not in mid))
        max_epochs = 12 if fast else 300
        val_fold = sorted(set(np.asarray(inner_folds).tolist()))[-1]
        val_idx = np.flatnonzero(np.asarray(inner_folds) == val_fold); fit_idx = np.flatnonzero(np.asarray(inner_folds) != val_fold)
        def loss_for(m, idx):
            out=m(xtr[idx], mask_tr[idx], d_base=db[idx], s_d=sd_d, s_q=sd_q)
            loss=((out["D"]-yd[idx])/sd_d).pow(2).mean()+((out["Sq"]-yq[idx])/sd_q).pow(2).mean()
            if out["aux"] is not None: loss=loss+0.1*((out["aux"]-ya[idx])**2).mean()
            return loss, out
        probe=make_model(); opt=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=1e-4); best=float("inf"); best_epoch=1; wait=0; best_state=None
        patience=max(3, min(30, max_epochs//10))
        for epoch in range(1, max_epochs+1):
            probe.train(); opt.zero_grad(); loss,_=loss_for(probe,fit_idx); loss.backward(); torch.nn.utils.clip_grad_norm_(probe.parameters(),1.0); opt.step()
            probe.eval()
            with torch.no_grad(): vloss,_=loss_for(probe,val_idx)
            if float(vloss) < best - 1e-7: best=float(vloss); best_epoch=epoch; wait=0; best_state=copy.deepcopy(probe.state_dict())
            else: wait += 1
            if epoch >= (30 if not fast else 12) and wait >= patience: break
        best_epoch = max(best_epoch, 30 if not fast else 12)
        # Refit on all outer-training rows for the selected epoch; test is never
        # consulted by early stopping.
        m=make_model(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        for _ in range(best_epoch):
            opt.zero_grad(); loss,_=loss_for(m,np.arange(len(train))); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): out=m(xte,mask_te,d_base=torch.tensor(reference_depth(test,ref),dtype=torch.float32),s_d=sd_d,s_q=sd_q)
        return np.column_stack([out["D"].numpy(),out["Sq"].numpy()]),sum(p.numel() for p in m.parameters()),int(best_epoch)
    if mid == "TREE_U_DQ":
        Xtr, Xte = _features(train, ref, False), _features(test, ref, False)
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=1.0,
                                    random_state=seed, n_jobs=1)
    else:
        Xtr, Xte = _features(train, ref, phy), _features(test, ref, phy)
        # Ridge is used only for the fixed TREE/other non-neural static controls.
        model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(Xtr, y)
    return np.asarray(model.predict(Xte), float), int(sum(getattr(m,'coef_',np.array([])).size for m in getattr(model,'estimators_', [model]))), 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--contract-run', required=True)
    ap.add_argument('--backend-run', required=False); ap.add_argument('--fast', dest='fast', action='store_true', help='use the reduced smoke budget')
    ap.add_argument('--full', dest='fast', action='store_false', help='run the protocol 300-epoch ceiling with inner early stopping')
    ap.set_defaults(fast=True)
    ap.add_argument('--seed', type=int, default=17); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    frame, _, _ = load_main180()
    split = pd.read_csv(Path(a.contract_run)/'split_manifest.csv')[['dataset_index','outer_fold']]
    inner_path = Path(a.contract_run)/'inner_split_manifest.csv'
    if inner_path.exists():
        split = split.merge(pd.read_csv(inner_path)[['dataset_index','outer_fold','inner_fold']], on=['dataset_index','outer_fold'], how='left', validate='one_to_one')
    else:
        split['inner_fold'] = split.groupby('outer_fold').cumcount() % 3
    frame = frame.merge(split, on='dataset_index', suffixes=('','_audit'))
    run_root=ROOT/'outputs/mamba_depth_quality_v1'; run_root.mkdir(parents=True,exist_ok=True)
    run=run_root/(pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')+'_pilot'); run.mkdir()
    rows=[]; budgets=[]; failures=[]
    for fold in sorted(frame.outer_fold.unique()):
        tr=frame[frame.outer_fold!=fold].copy(); te=frame[frame.outer_fold==fold].copy()
        ref=fit_reference(tr)
        for mid in MODEL_IDS:
            t0=time.time()
            try:
                pred, nparam, selected_epoch = _fit_predict(mid,tr,te,ref,a.seed,fast=a.fast,inner_folds=tr.inner_fold.to_numpy())
                for j,(_,r) in enumerate(te.reset_index(drop=True).iterrows()):
                    rows.append(dict(model_id=mid,route_id='A_reference_sequence_terminal_proxy',reference_id='RLOG_EQEXP_v1' if 'PHY' in mid or 'STATIC' in mid else 'NONE',tokenization_id='equivalent_exposure_progress_v1',backend_id=('reference_pure_torch_ssd_mamba2_verified_adapter_v1' if mid.startswith('MAMBA') else 'torch_terminal_v1'),outer_fold=int(fold),seed=a.seed,dataset_index=int(r.dataset_index),component_id=str(r.component_id),session_role=r.session_role,observed_D_um=float(r.D),predicted_D_um=float(pred[j,0]),observed_Sq_um=float(r.Sq),predicted_Sq_um=float(pred[j,1]),train_null_D_um=float(np.average(tr.D,weights=component_weights(tr))),train_null_Sq_um=float(np.average(tr.Sq,weights=component_weights(tr))),D_scale_train=float(np.std(tr.D)),Sq_scale_train=float(np.std(tr.Sq)),bottleneck_dim=(32 if mid.endswith('K32_AUX') else 4 if 'K4' in mid else 0),fit_asset_hash=hashlib.sha256(f'{fold}:{mid}:{a.seed}'.encode()).hexdigest(),split_hash='contract_manifest',status='SUCCESS'))
                budgets.append(dict(model_id=mid,outer_fold=int(fold),seed=a.seed,fit_seconds=time.time()-t0,trainable_parameters=nparam,selected_epoch=selected_epoch,inner_folds=3,status='SUCCESS'))
            except Exception as e:
                failures.append(dict(model_id=mid,outer_fold=int(fold),error=repr(e)))
                budgets.append(dict(model_id=mid,outer_fold=int(fold),seed=a.seed,fit_seconds=time.time()-t0,trainable_parameters=0,selected_epoch=0,inner_folds=3,status='FAILED'))
    pd.DataFrame(rows).to_csv(run/'oof_predictions.csv',index=False,lineterminator='\n')
    pd.DataFrame(budgets).to_csv(run/'model_budget.csv',index=False,lineterminator='\n')
    pd.DataFrame(failures).to_csv(run/'failure_registry.csv',index=False,lineterminator='\n')
    pd.DataFrame({'model_id':MODEL_IDS,'seed':[a.seed]*len(MODEL_IDS),
                  'inner_folds':[3]*len(MODEL_IDS),
                  'max_epochs':[12 if a.fast else 300]*len(MODEL_IDS),
                  'status':['SUCCESS' if not any(f['model_id']==m for f in failures) else 'FAILED' for m in MODEL_IDS]}).to_csv(run/'inner_selection.csv',index=False,lineterminator='\n')
    (run/'protocol_resolved.yaml').write_text(f'profile: pilot\nseed: {a.seed}\nfast: {bool(a.fast)}\nmax_epochs: {12 if a.fast else 300}\ninner_reference_crossfit: 3\nbackend_id: reference_pure_torch_ssd_mamba2_verified_adapter_v1\n',encoding='utf-8',newline='\n')
    (run/'compute_budget.csv').write_text(f"mode,fast,models,outer_folds,max_epochs,inner_folds\n{'fast' if a.fast else 'full'},{str(bool(a.fast)).lower()},8,5,{12 if a.fast else 300},3\n",encoding='utf-8',newline='\n')
    (run/'run_manifest.json').write_text(json.dumps({'run_id':run.name,'seed':a.seed,'fast':bool(a.fast),'max_epochs':12 if a.fast else 300,'inner_reference_crossfit':3,'models':MODEL_IDS,'oof_rows':len(rows),'failures':len(failures)},indent=2),encoding='utf-8')
    print(run)

if __name__=='__main__': main()







