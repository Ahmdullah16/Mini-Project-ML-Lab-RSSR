"""Diagrams of actual cat-v1 feature paths; never infer trained availability."""
MODULES={'M1':(1,0,0),'M2':(0,1,0),'M3':(0,0,1),'M4':(1,1,0),
         'M5':(1,0,1),'M6':(0,1,1),'M7':(1,1,1)}


def diagram(key):
    c,a,t=MODULES[key]
    lines=['digraph {rankdir=LR; node [shape=box,style=rounded];',
           'Input [label="LR: N×C×H×W"]; E [label="3×3 embedding: C→d"];',
           'R [label="3×3: d→Cs²; PixelShuffle(s)"]; Add [label="Add → SR: N×C×sH×sW"];',
           'Input -> E; Input -> Bicubic -> Add; R -> Add;']
    if c:lines+=['L [label="C: f + Conv(ReLU(Conv(f)))"]; E -> L;']
    if t:lines+=['T [label="T: window attention + feed-forward residuals"]; E -> T;']
    if c and t:
        lines+=['Fuse [label="Concat + 1×1 projection: 2d→d"]; L -> Fuse; T -> Fuse;'];last='Fuse'
    else:last='L' if c else 'T' if t else 'E'
    if a:lines+=['A [label="A: channel gate then spatial gate"];',last+' -> A;'];last='A'
    return '\n'.join(lines+[last+' -> R;','}'])
