"""Shows FFT-grid quantisation in rPPG-Toolbox evaluation. usage: python verify_quantization.py /path/to/rPPG-Toolbox"""
import numpy as np, sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')  # path to rPPG-Toolbox
from evaluation.post_process import _calculate_peak_hr, calculate_metric_per_video
fs=30; rng=np.random.default_rng(0)
rep=[]; gte=[]
for k in range(200):
    hr=rng.uniform(55,110); t=np.arange(10*fs)/fs
    lab=np.sin(2*np.pi*hr/60*t); pr=np.sin(2*np.pi*(hr+1.5)/60*t+1)
    g,p,_,_=calculate_metric_per_video(pr,lab,fs=fs,diff_flag=False)
    rep.append(abs(p-g)); gte.append(abs(g-hr))
rep=np.array(rep); gte=np.array(gte)
print("10s windows, TRUE pred error = 1.50 bpm -> reported MAE %.2f (%.0f%% of windows reported 0.00)"%(rep.mean(), 100*np.mean(rep<1e-9)))
print("10s windows, |toolbox-derived GT HR - true HR|: mean %.2f, max %.2f bpm"%(gte.mean(), gte.max()))
t=np.arange(30*fs)/fs; s=np.sin(2*np.pi*1.2*t)+0.15*rng.standard_normal(len(t))
print("Peak-HR of a 72-bpm sine + light noise (bandpassed inside toolbox):", round(calculate_metric_per_video(s,s,fs=fs,diff_flag=False,hr_method='Peak')[0],1))
print("Peak-HR raw (no filter) same signal:", round(_calculate_peak_hr(s,fs),1))
