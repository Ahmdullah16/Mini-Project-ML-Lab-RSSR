"""Model-only CPU timing; MAC/FLOP and native peak allocation not claimed."""
import argparse,json,time,statistics
import torch
from rssr.experiment import load_model
p=argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--size',type=int,default=64); p.add_argument('--repeats',type=int,default=20)
a=p.parse_args()
if not 8<=a.size<=256 or not 1<=a.repeats<=100: p.error('bounded size/repetitions required')
torch.set_num_threads(2); model,state=load_model(a.checkpoint); x=torch.rand(1,model.bands,a.size,a.size)
with torch.inference_mode():
    for _ in range(3): model(x)
    times=[]
    for _ in range(a.repeats):
        t=time.perf_counter(); model(x); times.append(time.perf_counter()-t)
print(json.dumps({'run_id':state['run_id'],'input_shape':list(x.shape),'device':'cpu','precision':'float32','threads':2,'warmup':3,'repetitions':a.repeats,'median_seconds':statistics.median(times),'min_seconds':min(times),'max_seconds':max(times),'parameters':sum(p.numel() for p in model.parameters()),'timing_scope':'model only, excludes IO and tiling','macs':None,'peak_memory_bytes':None},indent=2))
