"""Export existing TeX reports to editable Word equations, text and tables."""
import argparse
from pathlib import Path
import re
import subprocess
from docx import Document
from docx.shared import Inches, Pt, RGBColor


def export(source, output, pandoc, bibliography, csl):
    source, output = Path(source).resolve(), Path(output).resolve()
    text = source.read_text(encoding='utf-8')
    for _ in range(5):
        text = re.sub(r'\\input\{([^}]+)\}', lambda m:(source.parent/m[1]).read_text(encoding='utf-8'), text)
    text = text.replace(r'\sbox{\resulttablebox}{%', '').replace(r'\end{tabular}}',r'\end{tabular}')
    text = re.sub(r'\\ifdim[^\n]+', '', text)
    text = re.sub(r'\\author\{.*?\}\n(?=\\(?:date|begin))',r'\\author{Ahmdullah}',text,flags=re.S)
    text = text.replace(r'\begin{IEEEkeywords}',r'\paragraph{Keywords}').replace(r'\end{IEEEkeywords}','')
    def flow(match):
        labels = re.findall(r'\\node\[.*?\]\s*\([^)]*\)\s*\{(.*?)\};',match[0],re.S)
        if not labels: raise ValueError('Diagram could not be converted without losing its nodes')
        return '\n'+r'\begin{tabular}{l}'+'\n'+('\\\\\n'+r'$\downarrow$'+'\\\\\n').join(labels)+'\\\\\n'+r'\end{tabular}'+'\nBicubic skip: the input also feeds the final addition.\n'
    text = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',flow,text,flags=re.S)
    text = text.replace(r'\captionof{table}',r'\textbf')
    output.parent.mkdir(parents=True,exist_ok=True)
    portable=output.with_suffix('.tex')
    portable.write_text(text,encoding='utf-8',newline='\n')
    subprocess.run([pandoc,str(portable),'-f','latex','-t','docx','--citeproc',
                    '--bibliography='+bibliography,'--csl='+csl,'-o',str(output)],check=True)
    doc=Document(output)
    for style in doc.styles:
        if style.type==1:
            style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
    doc.styles['Normal'].font.size=Pt(11)
    for sec in doc.sections:
        sec.page_width=Inches(8.5);sec.page_height=Inches(11)
        sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(.75)
    for p in doc.paragraphs:
        if p.style.name=='Author':
            p.text='Ahmdullah | Roll No. 264102307\nEE5302L: Introduction to Machine Learning Lab\nIndian Institute of Technology Guwahati'
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs: p.paragraph_format.keep_with_next=True
    doc.save(output)
    print(f'{output}: {len(doc.tables)} editable tables')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','output','pandoc','bibliography','csl']:p.add_argument('--'+key,required=True)
    export(**vars(p.parse_args()))
