"""Generate source-faithful, editable SVG block diagrams for RSSR model families."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'architecture-diagrams'
MODELS = {
'M1': ('CNN', ['Local residual CNN']),
'M2': ('Attention', ['Channel attention: GAP → 1×1 reduce → ReLU → 1×1 expand → sigmoid → multiply', 'Spatial attention: channel mean + max → concatenate → 7×7 convolution → sigmoid → multiply']),
'M3': ('Transformer', ['Replicate-pad to window grid', 'Window tokens + learned position → LayerNorm → 2-head self-attention → residual add', 'LayerNorm → Linear(2d) → GELU → Linear(d) → residual add → unpad']),
'M4': ('CNN + Attention', ['Local residual CNN', 'Channel attention: GAP → 1×1 reduce/expand → sigmoid → multiply', 'Spatial attention: channel mean/max → concat → 7×7 convolution → sigmoid → multiply']),
'M5': ('CNN + Transformer', ['Parallel local path: f + Conv3×3 → ReLU → Conv3×3', 'Parallel context path: pad → window tokens + position → LayerNorm → 2-head attention + residual → LayerNorm → FFN + residual → unpad', 'Concatenate paths (2d channels) → 1×1 projection to d']),
'M6': ('Attention + Transformer', ['Window Transformer: pad → tokens + position → LayerNorm → 2-head attention + residual → LayerNorm → FFN + residual → unpad', 'Channel attention: GAP → 1×1 reduce/expand → sigmoid → multiply', 'Spatial attention: channel mean/max → concat → 7×7 convolution → sigmoid → multiply']),
'M7': ('CNN + Attention + Transformer', ['Parallel local path: f + Conv3×3 → ReLU → Conv3×3', 'Parallel window Transformer: pad → tokens + position → LayerNorm → 2-head attention + residual → LayerNorm → FFN + residual → unpad', 'Concatenate (2d) → 1×1 projection (d)', 'Channel gate: GAP → 1×1 reduce/expand → sigmoid → multiply', 'Spatial gate: channel mean/max → concat → 7×7 convolution → sigmoid → multiply']),
}

def svg(model, title, details):
    W, node_h, gap = 1120, 74, 22
    labels = ['Input LR: N × C × H × W', 'Embedding: Conv 3×3, C → d'] + details + ['Reconstruction Conv 3×3, d → C·s²', 'PixelShuffle(s): N × C × sH × sW', 'Add bicubic baseline Bₛ(y) → SR output']
    heights = [62, 68] + [node_h] * len(details) + [68, 62, 62]
    y = 82
    blocks=[]
    colors=['#e8f1fb','#e8f1fb']+['#fff0d9']*len(details)+['#e8f1fb','#e8f1fb','#dff3e5']
    for i,(label,h,color) in enumerate(zip(labels,heights,colors)):
        blocks.append(f'<rect x="110" y="{y}" width="900" height="{h}" rx="14" fill="{color}" stroke="#243b53" stroke-width="2"/>')
        lines=[]
        # wrap by characters, preserving internal operations as readable lines
        words=label.split(); line='';
        for word in words:
            if len(line)+len(word)>95: lines.append(line); line=word
            else: line=(line+' '+word).strip()
        if line: lines.append(line)
        start=y+h/2-(len(lines)-1)*11
        for j,t in enumerate(lines): blocks.append(f'<text x="560" y="{start+j*23:.0f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="19" fill="#102a43">{escape(t)}</text>')
        if i<len(labels)-1:
            y2=y+h
            blocks.append(f'<path d="M560 {y2+2} L560 {y2+gap-3}" stroke="#243b53" stroke-width="2.5" marker-end="url(#arrow)"/>')
        y+=h+gap
    # residual interpolation bypass to final addition
    final_top=sum(heights[:-1])+gap*(len(heights)-1)+82
    blocks.append(f'<path d="M1010 113 H1060 V{final_top+31} H1015" fill="none" stroke="#1b7f5a" stroke-width="3" stroke-dasharray="9 6" marker-end="url(#green)"/>')
    height=y+28
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">
<title id="title">{model}: {escape(title)} remote sensing super-resolution architecture</title><desc id="desc">Implemented computation graph including internal attention or Transformer operations, reconstruction, PixelShuffle, and bicubic residual addition. Shapes use N batch, C bands, d feature width, H and W input dimensions, s scale.</desc>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#243b53"/></marker><marker id="green" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#1b7f5a"/></marker></defs>
<rect width="100%" height="100%" fill="#ffffff"/><text x="560" y="42" text-anchor="middle" font-family="Arial,sans-serif" font-size="27" font-weight="bold" fill="#102a43">{model}: {escape(title)}</text>{''.join(blocks)}
<text x="560" y="{height-8}" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#486581">Solid arrows: learned feature path · dashed green arrow: bicubic skip · Diagram follows src/rssr/models.py</text></svg>'''

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for key,(title,steps) in MODELS.items():
        (OUT/f'{key.lower()}_{title.lower().replace(" + ","_").replace(" ","_")}.svg').write_text(svg(key,title,steps),encoding='utf-8')
    print(f'Generated {len(MODELS)} architecture SVGs in {OUT}')
