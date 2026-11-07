"""Technical computation graphs for implemented cat-v1 modules; no model execution."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.backends.backend_pdf import PdfPages
OUT=Path(__file__).resolve().parents[2]/'submission-technical-diagrams'
COL={'conv':'#dcecff','att':'#ffe7c0','trans':'#e9defa','op':'#dcf2e8','io':'#edf1f5'}
class Graph:
 def __init__(self,title,subtitle,w=18,h=10):
  self.fig,self.ax=plt.subplots(figsize=(w,h));self.ax.set(xlim=(0,18),ylim=(0,10));self.ax.axis('off');self.fig.patch.set_facecolor('white');self.nodes={}
  self.ax.text(.3,9.65,title,fontsize=20,weight='bold',color='#12263a');self.ax.text(.3,9.13,subtitle,fontsize=10,color='#44576a')
 def box(self,k,x,y,t,c='conv',w=2.15,h=.95):
  self.nodes[k]=(x,y,w,h);self.ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.04,rounding_size=.09',edgecolor='#34516d',facecolor=COL[c],lw=1.3));self.ax.text(x,y,t,ha='center',va='center',fontsize=9)
 def op(self,k,x,y,t):
  self.nodes[k]=(x,y,.44,.44);self.ax.add_patch(Circle((x,y),.22,color=COL['op'],ec='#34516d',lw=1.5));self.ax.text(x,y,t,ha='center',va='center',fontsize=15)
 def edge(self,a,b,sa='r',sb='l',label='',via=None,dash=False):
  def point(k,s):
   x,y,w,h=self.nodes[k];return {'r':(x+w/2,y),'l':(x-w/2,y),'t':(x,y+h/2),'b':(x,y-h/2)}[s]
  p,q=point(a,sa),point(b,sb);pts=[p]+(via or [])+[q]
  for i in range(len(pts)-1):
   self.ax.add_patch(FancyArrowPatch(pts[i],pts[i+1],arrowstyle='-|>' if i==len(pts)-2 else '-',mutation_scale=12,lw=1.4,color='#24546d',linestyle='--' if dash else '-'))
  if label:
   mid=pts[len(pts)//2];self.ax.text(mid[0],mid[1]+.17,label,fontsize=8,color='#24546d',ha='center',bbox=dict(fc='white',ec='none',pad=1))
 def note(self,text):self.ax.text(.3,.35,text,fontsize=9,color='#44576a',va='bottom')
 def save(self,name,pdf):
  self.fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');self.fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight');pdf.savefig(self.fig,bbox_inches='tight');plt.close(self.fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with PdfPages(OUT/'RSSR_Technical_Block_Diagrams.pdf') as pdf:
  families=[('CNN',1,0,0),('Attention',0,1,0),('Transformer',0,0,1),('CNN + Attention',1,1,0),('CNN + Transformer',1,0,1),('Attention + Transformer',0,1,1),('CNN + Attention + Transformer',1,1,1)]
  for i,(name,c,a,t) in enumerate(families,1):
   g=Graph(f'M{i} | {name}',f'cat-v1/M{i} · actual feature graph · modules expanded in the accompanying internal schematics')
   g.box('x',1.4,6,'LR input y\nN × C × H × W','io');g.box('e',4,6,'Embedding E\nConv2d C → d\n3×3, stride 1, pad 1');g.edge('x','e')
   if c and t:
    g.box('c',7.1,7.5,'C: local residual\nL(f) = f + C(f)');g.box('t',7.1,4.5,'T: window encoder\nT(f)','trans');g.edge('e','c',via=[(5.4,6),(5.4,7.5)]);g.edge('e','t',via=[(5.4,6),(5.4,4.5)])
    g.box('f',10.2,6,'Concat channels [L;T]\nN × 2d × H × W\nConv1×1: 2d → d','op');g.edge('c','f',sb='t',via=[(10.2,7.5)]);g.edge('t','f',sb='b',via=[(10.2,4.5)]);last='f'
   else:
    last='e'
    if c:g.box('c',7.1,6,'C: local residual\nL(f) = f + C(f)');g.edge(last,'c');last='c'
    if t:g.box('t',7.1,6,'T: window encoder\nMHSA + residual FFN','trans');g.edge(last,'t');last='t'
   if a:g.box('a',13.2,6,'A: channel × spatial\nMultiplicative feature gates','att');g.edge(last,'a');last='a'
   g.box('r',15.9,3,'Reconstruction\nConv3×3: d → Cs²\nPixelShuffle(s)');g.edge(last,'r',sa='b',sb='t',via=[(g.nodes[last][0],3.9),(15.9,3.9)])
   g.op('sum',12.5,1.7,'+');g.edge('r','sum',sa='b',sb='r',via=[(15.9,1.7)])
   g.box('b',5,1.7,'Bicubic Bₛ(y)\nalign_corners=False','io');g.edge('x','b',sa='b',via=[(1.4,1.7)],dash=True);g.edge('b','sum',sa='b',sb='b',via=[(5,.9),(12.5,.9)],dash=True)
   g.box('o',9,1.7,'SR prediction\nx̂ = Bₛ(y) + residual','io');g.edge('sum','o',sa='l',sb='r')
   g.note('N: batch · C: configured bands · d: feature width · s: native scale (2 or 4) · all backbone features retain H × W.\nNo extra activation follows the embedding, reconstruction convolution or final addition. These diagrams describe operations, not measured performance.')
   g.save(f'M{i}_technical',pdf)
  g=Graph('C | Residual local convolution block','Exact implemented sequence: two 3×3 convolutions with a single intervening ReLU')
  for k,x,t in [('f',1.6,'Feature f\nN × d × H × W'),('c1',5,'Conv2d d → d\n3×3, stride 1, pad 1'),('relu',8,'ReLU\nmax(0,z)'),('c2',11,'Conv2d d → d\n3×3, stride 1, pad 1')]:g.box(k,x,6,t)
  g.op('s',14,6,'+');g.box('out',16.3,6,'L(f) = f + C(f)')
  for a,b in [('f','c1'),('c1','relu'),('relu','c2'),('c2','s'),('s','out')]:g.edge(a,b)
  g.edge('f','s',sa='t',sb='t',via=[(1.6,8),(14,8)],label='identity residual')
  g.note('Each convolution has 9d² + d trainable parameters (bias included). Total C block: 18d² + 2d.\nThe identity shortcut carries the original feature tensor; addition requires matching shapes. Zero padding is used by these convolutions.')
  g.save('C_residual_convolution',pdf)
  g=Graph('A | Channel and spatial attention gates','Channel gating is applied first; spatial statistics are computed from the channel-gated features')
  g.box('f',1.5,7,'Feature f\nN × d × H × W');g.box('gap',4.3,7,'AdaptiveAvgPool2d(1)\nN × d × 1 × 1','att');g.box('mlp',7.4,7,'Conv1×1 d → r\nReLU\nConv1×1 r → d','att');g.box('sig',10.4,7,'Sigmoid σ\nchannel weights','att');g.op('m',13,7,'×');g.box('fc',15.8,7,'f_c = f ⊙ channel(f)')
  for a,b in [('f','gap'),('gap','mlp'),('mlp','sig'),('sig','m'),('m','fc')]:g.edge(a,b)
  g.edge('f','m',sa='t',sb='t',via=[(1.5,8.3),(13,8.3)],label='broadcast multiplication')
  g.box('mean',3,4.2,'Mean over channels\nN × 1 × H × W','att');g.box('max',3,2.7,'Max over channels\nN × 1 × H × W','att');g.edge('fc','mean',sa='b',sb='t',via=[(15.8,5.5),(3,5.5)]);g.edge('fc','max',sa='b',sb='l',via=[(15.8,5.1),(1.6,5.1),(1.6,2.7)])
  g.box('cat',6.2,3.5,'Concatenate\nN × 2 × H × W','att');g.edge('mean','cat',sb='t',via=[(6.2,4.2)]);g.edge('max','cat',sb='b',via=[(6.2,2.7)])
  g.box('sp',9.3,3.5,'Conv7×7: 2 → 1\npadding 3 → sigmoid','att');g.op('m2',12.5,3.5,'×');g.box('o',15.4,3.5,'A(f) = f_c ⊙ spatial(f_c)');g.edge('cat','sp');g.edge('sp','m2');g.edge('m2','o');g.edge('fc','m2',sa='b',sb='t',via=[(15.8,5),(12.5,5)],label='feature bypass')
  g.note('r = max(1, floor(d/4)). Channel weights broadcast across H,W; spatial weights broadcast across d.\nGlobal pooling makes attention tile-context dependent. Gates weight existing features; these are not neuron connectivity or physical-confidence maps.')
  g.save('A_channel_spatial_gates',pdf)
  g=Graph('T | Window Transformer with two residual connections','Exact position placement: q = LayerNorm(z + P); attention residual adds original z, not z + P')
  for k,x,y,t in [('f',1.6,7,'Feature f\nN × d × H × W'),('pad',4.5,7,'Replicate pad\nH′,W′ divisible by w'),('tok',7.5,7,'Partition windows\nz: Bᵥ × w² × d'),('pos',10.6,7,'z + learned P\nP: 1 × w² × d'),('ln',13.6,7,'LayerNorm(d)\nq = LN(z + P)')]:g.box(k,x,y,t,'trans')
  for a,b in [('f','pad'),('pad','tok'),('tok','pos'),('pos','ln')]:g.edge(a,b)
  g.box('mh',13.6,4.5,'MultiheadAttention\nQ=K=V=q; 2 heads','trans');g.edge('ln','mh',sa='b',sb='t');g.op('s1',10.5,4.5,'+');g.edge('mh','s1',sa='l',sb='r');g.edge('tok','s1',sa='b',sb='t',via=[(7.5,5.5),(10.5,5.5)],label='original z')
  g.box('ff',7.4,4.5,'LN(d) → Linear(d,2d)\nGELU → Linear(2d,d)','trans');g.edge('s1','ff',sa='l',sb='r');g.op('s2',4.3,4.5,'+');g.edge('ff','s2',sa='l',sb='r');g.edge('s1','s2',sa='b',sb='b',via=[(10.5,3.2),(4.3,3.2)],label='post-attention residual')
  g.box('merge',4.3,1.8,'Reverse windows\nN × d × H′ × W′','trans');g.box('crop',8,1.8,'Remove only padding\nN × d × H × W','trans');g.edge('s2','merge',sa='b',sb='t');g.edge('merge','crop')
  g.note('Bᵥ = N(H′/w)(W′/w). d must be even; window w ≥ 1. No shifted windows, attention dropout or extra decoder is implemented.\nFixed-window dense attention has w² tokens per window. Replicate padding is removed before reconstruction.')
  g.save('T_window_encoder',pdf)
  g=Graph('MHSA | Two-head scaled dot-product self-attention','Internal operation of torch.nn.MultiheadAttention(width, 2, batch_first=True, dropout=0)')
  g.box('q',1.6,6,'Normalized tokens q\nBᵥ × L × d','trans')
  for k,y,t in [('Q',8,'Linear WQ + bQ\nQ: Bᵥ × 2 × L × dₕ'),('K',6,'Linear WK + bK\nK: Bᵥ × 2 × L × dₕ'),('V',4,'Linear WV + bV\nV: Bᵥ × 2 × L × dₕ')]:g.box(k,5,y,t,'trans');g.edge('q',k,via=[(3,6),(3,y)])
  g.box('score',8.5,7,'QKᵀ / √dₕ\nper-head score matrix','op');g.edge('Q','score',sb='t',via=[(8.5,8)]);g.edge('K','score',sb='b',via=[(8.5,6)])
  g.box('soft',11.6,7,'Softmax over keys\nA: Bᵥ × 2 × L × L','att');g.edge('score','soft');g.op('mul',11.6,4,'×');g.edge('soft','mul',sa='b',sb='t');g.edge('V','mul',label='AV per head')
  g.box('cat',15,4,'Concat two heads\nLinear WO + bO\nBᵥ × L × d','trans');g.edge('mul','cat')
  g.note('L = w²; dₕ = d/2. Aᵢⱼ = exp(scoreᵢⱼ) / Σₖ exp(scoreᵢₖ), so each query row sums to 1.\nProjections are trainable and generally mix all feature channels. No causal mask; windows provide the spatial connectivity restriction.\nDense attention-score memory is proportional to Bᵥ · 2 · L² elements; actual runtime memory also includes projections and activations.')
  g.save('MHSA_two_head_operations',pdf)
  g=Graph('R | Reconstruction and sub-pixel rearrangement','The output is a learned residual added to bicubic interpolation; no final clamp occurs in SRModel.forward')
  for k,x,t in [('f',1.6,'Backbone feature\nN × d × H × W'),('conv',5,'Conv3×3 d → Cs²\nN × Cs² × H × W'),('ps',8.5,'PixelShuffle(s)\nchannel → spatial indices')]:g.box(k,x,6,t)
  g.op('sum',12.3,6,'+');g.box('out',15.3,6,'SR output\nN × C × sH × sW','io')
  for a,b in [('f','conv'),('conv','ps'),('ps','sum'),('sum','out')]:g.edge(a,b)
  g.box('x',5,3,'Original LR y','io');g.box('b',9,3,'Bicubic interpolation\nscale_factor=s\nalign_corners=False','io');g.edge('x','b',dash=True);g.edge('b','sum',sb='b',via=[(12.3,3)],dash=True)
  g.ax.text(.7,1.5,'PixelShuffle: output[n,c,sh+i,sw+j] = input[n,c·s²+i·s+j,h,w],  0 ≤ i,j < s.',fontsize=13)
  g.note('Reconstruction convolution: 9dCs² + Cs² parameters. PixelShuffle itself has no trainable parameters.\nCascades execute this native module repeatedly; scale multiplies each dimension and pixel count grows quadratically. Enlargement does not prove detail recovery.')
  g.save('R_pixelshuffle_reconstruction',pdf)
 print('Created 12 technical sheets: 7 full models + 5 internal modules; PNG, SVG, and combined PDF.')
if __name__=='__main__':main()