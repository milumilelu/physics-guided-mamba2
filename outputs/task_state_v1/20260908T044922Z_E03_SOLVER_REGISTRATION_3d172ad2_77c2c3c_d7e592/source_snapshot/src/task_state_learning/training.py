"""Joint terminal/DP/closure optimization with atomic epoch checkpoints."""
import math,json,hashlib,os
from pathlib import Path
import numpy as np
import torch
from .grouping import require
from .losses import decision_gap_loss,decision_gap_scale,terminal_scale,PairSampler,LAMBDA_DP,LAMBDA_CLOSURE

WEIGHT_DECAY=1e-4
GRAD_CLIP=1.
BATCH=4
ACCUM=4
MAX_EPOCHS=300
PATIENCE=40
EVAL_CHUNK=64


def _batches(n,batch,generator):
    order=torch.randperm(n,generator=generator)
    return list(order.split(batch))


def _forward_chunks(model,control,dt,mask,chunk=EVAL_CHUNK):
    return torch.cat([model(control[i:i+chunk],dt[i:i+chunk],mask[i:i+chunk])
                      for i in range(0,len(control),chunk)])


def _cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _atomic_save(value,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def fit(model,train,val,*,lr,seed,device='cpu',max_epochs=MAX_EPOCHS,
        patience=PATIENCE,batch=BATCH,accum=ACCUM,dp=None,fixed_epochs=None,
        return_history=False,checkpoint_dir=None,resume=True,epoch_callback=None):
    require(lr>0 and batch>=1 and accum>=1,'Invalid optimization budget')
    torch.manual_seed(seed)
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WEIGHT_DECAY)
    tensors={k:v.to(device) for k,v in train.items()}
    validation={k:v.to(device) for k,v in val.items()}
    scale=terminal_scale(tensors['terminal_x']).detach()
    generator=torch.Generator().manual_seed(seed)
    total_epochs=fixed_epochs if fixed_epochs is not None else max_epochs
    require(total_epochs>=1,'Epoch budget must be positive')
    anchors=dp['anchors'].to(device) if dp else None
    gap_scale=decision_gap_scale(tensors['terminal_x'],anchors,scale) if dp else None
    signature={'lr':lr,'seed':seed,'batch':batch,'accum':accum,'epochs':total_epochs,
               'fixed_epochs':fixed_epochs,'patience':patience,'model':type(model).__name__,
               'shapes':{k:list(v.shape) for k,v in model.state_dict().items()},
               'dp':None if dp is None else {'seed':dp['sampler'].seed,'pairs':dp['sampler'].pairs}}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode())
    for source in ('training.py','models.py','memory.py','mamba2.py','losses.py','static_baseline.py'):
        digest.update(Path(__file__).with_name(source).read_bytes())
    for data in (train,val):
        for key in sorted(data):digest.update(data[key].cpu().contiguous().numpy().tobytes())
    if anchors is not None:digest.update(anchors.detach().cpu().numpy().tobytes())
    fingerprint=digest.hexdigest()
    checkpoint=None
    if checkpoint_dir is not None:
        directory=Path(checkpoint_dir);directory.mkdir(parents=True,exist_ok=True)
        checkpoint=directory/'checkpoint.pt'
    best={'epoch':-1,'val_loss':math.inf,'state':None}
    history=[];start_epoch=0
    if checkpoint is not None and checkpoint.exists():
        require(resume,'Existing checkpoint requires resume')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(saved['fingerprint']==fingerprint,'Checkpoint configuration/data/source mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best=saved['best'];history=saved['history'];start_epoch=saved['next_epoch']
        generator.set_state(saved['generator_state']);torch.set_rng_state(saved['torch_rng'])
        if device.startswith('cuda') and saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for epoch in range(start_epoch,total_epochs):
        if fixed_epochs is None and history and history[-1]['epoch']-best['epoch']>=patience:break
        model.train()
        batches=_batches(len(tensors['control']),batch,generator)
        groups=[batches[i:i+accum] for i in range(0,len(batches),accum)]
        pair_groups=[None]*len(groups)
        if dp:
            sampler=PairSampler(len(tensors['control']),dp['sampler'].pairs,dp['sampler'].seed+epoch)
            pair_groups=list(torch.tensor_split(sampler.sample().to(device),len(groups)))
        epoch_loss=0.
        for microbatches,pairs in zip(groups,pair_groups):
            optimizer.zero_grad()
            n=sum(len(indices) for indices in microbatches)
            for indices in microbatches:
                sub={k:v[indices] for k,v in tensors.items()}
                if hasattr(model,'readout'):
                    pred,closure=model(sub['control'],sub['physical_dt'],sub['mask'],return_closure=True)
                else:
                    pred=model(sub['control'],sub['physical_dt'],sub['mask']);closure=pred.new_zeros(())
                loss=((pred-sub['terminal_x'])/scale).pow(2).mean()+LAMBDA_CLOSURE*closure
                weighted=loss*(len(indices)/n)
                weighted.backward();epoch_loss+=float(weighted.detach())/len(groups)
            if pairs is not None and len(pairs):
                unique,inverse=torch.unique(pairs.reshape(-1),sorted=True,return_inverse=True)
                pred=model(tensors['control'][unique],tensors['physical_dt'][unique],tensors['mask'][unique])
                loss=decision_gap_loss(pred,tensors['terminal_x'][unique],anchors,scale,
                                       inverse.reshape(-1,2),gap_scale)
                # Weight uneven pair partitions so the epoch mean retains equal pair mass.
                weighted=LAMBDA_DP*loss*(len(pairs)*len(groups)/dp['sampler'].pairs)
                weighted.backward();epoch_loss+=float(weighted.detach())/len(groups)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP,error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction=_forward_chunks(model,validation['control'],validation['physical_dt'],validation['mask'])
            val_loss=float(((prediction-validation['terminal_x'])/scale).pow(2).mean())
        require(math.isfinite(val_loss),'Nonfinite validation loss')
        history.append({'epoch':epoch,'epochs_completed':epoch+1,'val_loss':val_loss,'train_objective':epoch_loss})
        if fixed_epochs is None and val_loss<best['val_loss']:
            best={'epoch':epoch,'val_loss':val_loss,'state':_cpu_state(model)}
        if checkpoint is not None:
            _atomic_save({'fingerprint':fingerprint,'signature':signature,'model':_cpu_state(model),
                          'optimizer':optimizer.state_dict(),'best':best,'history':history,'next_epoch':epoch+1,
                          'generator_state':generator.get_state(),'torch_rng':torch.get_rng_state(),
                          'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else []},checkpoint)
            log=checkpoint.parent/'history.json';tmp=log.with_suffix('.tmp')
            tmp.write_text(json.dumps(history,allow_nan=False),encoding='utf-8');os.replace(tmp,log)
        if epoch_callback:epoch_callback(history[-1])
        if fixed_epochs is None and epoch-best['epoch']>=patience:break
    if fixed_epochs is None:
        require(best['state'] is not None,'Training produced no evaluation point')
        model.load_state_dict(best['state']);selected=best['epoch'];value=best['val_loss']
    else:selected=total_epochs-1;value=history[-1]['val_loss']
    model=model.cpu()
    if checkpoint is not None:_atomic_save(_cpu_state(model),checkpoint.parent/'selected_state.pt')
    return {'model':model,'best_epoch':selected,'selected_epochs':selected+1,'val_loss':value,
            'history':history if return_history else None,'fingerprint':fingerprint}


def predict(model,data,device='cpu',batch=EVAL_CHUNK):
    model=model.to(device).eval()
    with torch.no_grad():
        pred=_forward_chunks(model,data['control'].to(device),data['physical_dt'].to(device),
                             data['mask'].to(device),batch).cpu()
    model.cpu()
    return pred
