"""numpy port of neural_methods/loss/NegPearsonLoss.py (dim=0) vs temporal Pearson."""
import numpy as np
rng=np.random.default_rng(1)
def negp_dim0(p,l):  # exact numpy port of neural_methods/loss/NegPearsonLoss.py (cos over dim=0)
    pc=p-p.mean(0,keepdims=True); lc=l-l.mean(0,keepdims=True)
    cos=(pc*lc).sum(0)/np.maximum(np.linalg.norm(pc,axis=0)*np.linalg.norm(lc,axis=0),1e-6)
    return np.mean(1-cos)
def negp_time(p,l):
    pc=p-p.mean(1,keepdims=True); lc=l-l.mean(1,keepdims=True)
    return np.mean(1-(pc*lc).sum(1)/np.linalg.norm(pc,axis=1)/np.linalg.norm(lc,axis=1))
T=160; t=np.arange(T)/30
lab=np.stack([np.sin(2*np.pi*f*t) for f in [1.0,1.3,1.6,2.0]])
good=lab.copy()
shifted=np.stack([np.sin(2*np.pi*f*t+0.8) for f in [1.0,1.3,1.6,2.0]])   # right HR, phase lag
wrongHR=np.stack([np.sin(2*np.pi*f*t) for f in [1.6,2.0,1.0,1.3]])       # HR swapped across samples
print("B=4  perfect   : dim0 %.3f | temporal %.3f"%(negp_dim0(good,lab),negp_time(good,lab)))
print("B=4  phase lag : dim0 %.3f | temporal %.3f"%(negp_dim0(shifted,lab),negp_time(shifted,lab)))
print("B=4  wrong HRs : dim0 %.3f | temporal %.3f"%(negp_dim0(wrongHR,lab),negp_time(wrongHR,lab)))
print("B=1  any pred  : dim0 %.3f (constant -> zero gradient)"%negp_dim0(rng.standard_normal((1,T)),lab[:1]))
