"""Independent B1 float64 reference and terminal-only data contract.

No laser solver or recurrent learner is used by this reference. Control jumps
are integration boundaries; hidden states are exported only to evaluator files.
"""
import numpy as np
from .grouping import require


def b1_rhs(state, control, coupling=True):
    x1,x2,q1,q2=np.moveaxis(np.asarray(state,dtype=np.float64),-1,0)
    u1,u2=np.moveaxis(np.asarray(control,dtype=np.float64),-1,0)
    return np.stack((-0.4*x1+u1+(0.6*q1-0.2*q2 if coupling else 0.),
                     -0.7*x2+0.5*u2+0.1*x1*u1+(0.3*np.tanh(q2) if coupling else 0.),
                     (-q1+u1)/.05,(-q2+u1*u1+.5*u2)/.5),axis=-1)


def b1_reference(controls,block_duration,*,coupling=True,max_step=.001,initial=None,return_blocks=False):
    controls=np.asarray(controls,dtype=np.float64)
    duration=np.asarray(block_duration,dtype=np.float64)
    require(controls.ndim==3 and controls.shape[-1]==2,'B1 controls require families x blocks x 2')
    require(np.isfinite(controls).all() and (np.abs(controls)<=1).all(),'Invalid B1 controls')
    require(duration.shape==controls.shape[:2] and np.isfinite(duration).all() and (duration>0).all(),'Invalid B1 durations')
    require(0<max_step<=.001,'B1 reference step must not exceed 0.001')
    state=np.zeros((len(controls),4)) if initial is None else np.array(initial,dtype=np.float64,copy=True)
    require(state.shape==(len(controls),4) and np.isfinite(state).all(),'Invalid B1 initial state')
    history=[]
    for block in range(controls.shape[1]):
        count=np.ceil(duration[:,block]/max_step).astype(int)
        h=(duration[:,block]/count)[:,None]
        u=controls[:,block]
        for step in range(int(count.max())):
            k1=b1_rhs(state,u,coupling)
            k2=b1_rhs(state+h*k1/2,u,coupling)
            k3=b1_rhs(state+h*k2/2,u,coupling)
            k4=b1_rhs(state+h*k3,u,coupling)
            candidate=state+h*(k1+2*k2+2*k3+k4)/6
            state=np.where((step<count)[:,None],candidate,state)
        history.append(state.copy())
    require(np.isfinite(state).all(),'Nonfinite B1 reference')
    return (state,np.stack(history,axis=1)) if return_blocks else state


def sample_b1_families(count,seed_sequence,*,lengths=(32,64,128),dt_range=(.01,.08)):
    rng=np.random.default_rng(seed_sequence)
    controls=rng.uniform(-1,1,(count,8,2))
    k=rng.choice(lengths,count)
    dt=rng.uniform(*dt_range,count)
    duration=np.repeat((k*dt/8)[:,None],8,axis=1)
    return {'controls':controls,'token_count':k,'token_dt':dt,'block_duration':duration}


def terminal_arrays(families,terminal,family_ids):
    k=families['token_count'];maximum=int(k.max())
    u=np.zeros((len(k),maximum,2));dt=np.zeros((len(k),maximum));mask=np.zeros((len(k),maximum),bool)
    for i,n in enumerate(k):
        require(n%8==0,'B1 token count must preserve eight control blocks')
        u[i,:n]=np.repeat(families['controls'][i],n//8,axis=0)
        dt[i,:n]=families['token_dt'][i];mask[i,:n]=True
    return {'control':u,'physical_dt':dt,'mask':mask,'terminal_x':np.asarray(terminal)[:,:2],
            'family_id':np.asarray(family_ids,dtype='U64')}


def load_terminal_only(path):
    expected={'control','physical_dt','mask','terminal_x','family_id'}
    with np.load(path,allow_pickle=False) as data:
        require(set(data.files)==expected,'Terminal-only file contains forbidden or missing arrays')
        values={k:data[k] for k in expected}
    require(values['terminal_x'].shape==(len(values['family_id']),2),'Terminal label shape mismatch')
    require(values['control'].shape[:2]==values['mask'].shape==values['physical_dt'].shape,'Token shape mismatch')
    require(len(np.unique(values['family_id']))==len(values['family_id']),'Duplicate base history family')
    require(np.isfinite(values['control']).all() and np.isfinite(values['terminal_x']).all(),'Nonfinite terminal-only arrays')
    require((values['physical_dt'][values['mask']]>0).all(),'Invalid physical dt')
    return values
