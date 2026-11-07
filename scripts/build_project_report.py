"""Typeset existing teaching derivations and frozen evidence; no new experiments."""
import argparse
import hashlib
import json
import csv
import sys
import re
import subprocess
from pathlib import Path
import pypandoc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
SOURCES = ['docs/student-project-guide.md', 'docs/learning-reference.md',
           'docs/model-instructor.md', 'docs/metric-instructor.md']


def build(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures=output/'figures'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_technical_architectures.py'),'--output',str(figures)],check=True)
    parts = [r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx,calc,caption}
\setcounter{tocdepth}{1}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{Consolas}
\usepackage[hidelinks]{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newsavebox{\resulttablebox}
\title{Attention- and Transformer-Based Deep Learning for Super-Resolution of Remote Sensing Images\\Detailed Project Report}
\author{Ahmdullah\\Roll No. 264102307\\EE5302L: Introduction to Machine Learning Lab\\Indian Institute of Technology Guwahati}
\date{8 November 2026}
\begin{document}\sloppy\maketitle
\begin{abstract}
This report explains the implemented educational architecture family and the recorded remote-sensing super-resolution experiments. It reuses existing mathematical teaching material and immutable experimental tables. Historical studies have different bands and protocols and must not be combined into a seven-model ranking. Synthetic reconstruction measurements do not demonstrate calibrated cross-sensor transfer or recovery of physically missing details. Software capabilities, observed measurements and remaining scientific questions are distinguished throughout.
\end{abstract}
\tableofcontents\clearpage
''']
    hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'src/rssr/models.py',ROOT/'src/rssr/architecture_registry.py',ROOT/'scripts/generate_technical_architectures.py']}
    for source in SOURCES:
        text = (ROOT/source).read_text(encoding='utf-8-sig')
        hashes[source] = hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        text = text.translate(str.maketrans({'─':'-', '├':'+', '┤':'+', '┐':'+', '┘':'+', '│':'|', '└':'+'}))
        text = text.replace('→', ' → ').replace('×', ' × ')
        text = re.sub(r',(?=[A-Za-z])', ', ', text)
        text = re.sub(r'-{8,}', '------', text)
        text = re.sub(r'```text\nlow-resolution image.*?```',
                      '```text\nLR -> bicubic resize ------------------+\nLR -> embedding -> blocks -> residual -+-> add -> SR\n```', text, flags=re.S)
        parts.append(pypandoc.convert_text(text, 'latex', format='markdown+tex_math_single_backslash',
                                          extra_args=['--top-level-division=section', '--syntax-highlighting=none']))
        parts.append(r'\clearpage')
    literature = (ROOT/'paper/literature-context.tex').read_text(encoding='utf-8')
    parts.extend([r'\section{Literature context}', literature])
    hashes['paper/literature-context.tex'] = hashlib.sha256(literature.encode()).hexdigest()
    bibliography = (ROOT/'references.bib').read_bytes()
    (output/'references.bib').write_bytes(bibliography)
    hashes['references.bib'] = hashlib.sha256(bibliography).hexdigest()
    parts.append(r'''\section{Seven architecture paths}
Let $f=E(y)$, $L(f)=f+C(f)$, $P(u,v)=W_f*[u;v]+b_f$, and $A$ and $T$ denote the attention and window Transformer blocks explained above. Each path predicts $\hat x=B_s(y)+R_s(G_i(f))$. These are definitions of implemented computations, not performance rankings. Shared convolutions remain present in every family.
''')
    paths = [
        ('M1', 'CNN', r'L(f)', 'Local residual CNN'),
        ('M2', 'Attention', r'A(f)', 'Channel then spatial attention'),
        ('M3', 'Transformer', r'T(f)', 'Window Transformer'),
        ('M4', 'CNN + Attention', r'A(L(f))', 'Local residual CNN then attention'),
        ('M5', 'CNN + Transformer', r'P(L(f),T(f))', 'Parallel CNN and Transformer; concat + projection'),
        ('M6', 'Attention + Transformer', r'A(T(f))', 'Window Transformer then feature attention'),
        ('M7', 'CNN + Attention + Transformer', r'A(P(L(f),T(f)))', 'Parallel CNN and Transformer; fusion then attention'),
    ]
    from rssr.architecture_registry import describe
    counts = [describe('cat-v1/'+item[0]) for item in paths]
    with (output/'RSSR_Parameters.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['architecture_id', 'bands', 'width', 'window', 'native_scale', 'parameters', 'scope'])
        for row in counts:
            writer.writerow([row['architecture_id'],3,16,4,2,row['parameters'],'Untrained registry configuration; no weights loaded'])
    parts.append(r'\begin{center}\begin{tabular}{lr}\toprule Architecture & Parameters\\\midrule')
    parts.extend(row['architecture_id'] + ' & ' + str(row['parameters']) + r'\\' for row in counts)
    parts.append(r'\bottomrule\end{tabular}\end{center}')
    parts.append('Counts are computed from instantiated untrained registry modules at RGB bands 3, width 16, window 4 and native scale 2. They are not checkpoint-loading or accuracy evidence.\n')
    parts.append(r'\clearpage')
    for ident, title, equation, body in paths:
        parts.append(r'\subsection{' + ident + ': ' + title + '}')
        parts.append(r'\[G_{' + ident[1:] + '}(f)=' + equation + r'\]')
        parts.append(r'\begin{center}\includegraphics[width=\linewidth]{figures/' + ident + r'_technical.png}\end{center}')
        parts.append('The connections show the implemented tensor operations. Read the following internal schematics for kernel sizes, gating, normalization and residual paths. Native scale and bands must match the selected checkpoint.\n')
        parts.append(r'\clearpage')
    parts.append(r'\section{Internal module operations and connections}')
    for filename, title in [
        ('C_residual_convolution', 'Residual local convolution'),
        ('A_channel_spatial_gates', 'Channel and spatial feature gates'),
        ('T_window_encoder', 'Window Transformer encoder'),
        ('MHSA_two_head_operations', 'Multi-head attention projections and mixing'),
        ('R_pixelshuffle_reconstruction', 'Sub-pixel reconstruction and residual output')]:
        parts.append(r'\subsection{' + title + '}')
        parts.append(r'\begin{center}\includegraphics[width=\linewidth]{figures/' + filename + r'.png}\end{center}\clearpage')
    parts.append(r'\section{Recorded experimental results}')
    parts.append('The following tables are copied from the verified evidence generator. Seed variation is not geographic uncertainty. No new inference was run to build this report.\n')
    for source in ['paper/generated/results.tex', 'paper/generated/interpretation.tex']:
        data = (ROOT/source).read_bytes()
        hashes[source] = hashlib.sha256(data).hexdigest()
        table_text = re.sub(r'\\begin\{table\}(?:\[[^]]*\])?', r'\\begin{center}', data.decode('utf-8'))
        table_text = table_text.replace(r'\end{table}', r'\end{center}').replace(r'\caption{', r'\captionof{table}{')
        parts.append(table_text)
    parts.append(r'''\clearpage\section{Submission and scientific scope}
The available frozen evaluation contains one held-out scene. Matched-budget geographic comparisons, eligible calibrated cross-sensor evaluation and a complete reference-based failure atlas remain open. The code supports seven architecture families, but this report does not infer equal training budgets or architectural superiority from their availability. The IEEE manuscript and detailed forty-project supplement remain separate companion sources. A final reference audit and visual review are required before submission.
\bibliographystyle{plain}\bibliography{references}
\end{document}''')
    target = output/'RSSR_Final_Project_Report.tex'
    tex_source = '\n\n'.join(parts).replace('\r\n', '\n').replace(r'\def\LTcaptype{none}', '')
    target.write_text(tex_source, encoding='utf-8', newline='\n')
    (output/'report-source-manifest.json').write_text(json.dumps({
        'kind': 'derived report; no new measurements', 'sources_sha256': hashes,
        'tex_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'visual_review': 'pending'}, indent=2), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    build(parser.parse_args().output)
