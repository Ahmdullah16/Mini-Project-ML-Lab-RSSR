"""Shared presentation only: no inference, acquisition, account or training side effects."""
from html import escape
import streamlit as st

SECTION_LABELS = {
    'Overview': 'Overview', 'Live SR': 'AI Studio', 'Results': 'Results & benchmarks',
    'Methodology & Data': 'Model Lab & methodology', 'Mini-projects': 'Learning Lab',
    'Satellite Explorer': 'Satellite Explorer', 'Downloads & Reproducibility': 'Research & downloads',
}
PALETTES = {
    'Light': dict(bg='#F5F8FC', card='#FFFFFF', text='#14253D', muted='#4C617A', line='#CBD7E6', accent='#086B83', soft='#E5F3F7'),
    'Dark': dict(bg='#0B1220', card='#172339', text='#F8FAFC', muted='#B8C6DB', line='#425675', accent='#67E8F9', soft='#20384B'),
}


def apply_theme(mode):
    p = PALETTES[mode]
    st.markdown('''<style>
:root{--lab-bg:BG;--lab-card:CARD;--lab-text:TEXT;--lab-muted:MUTED;--lab-line:LINE;--lab-accent:ACCENT;--lab-soft:SOFT;}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--lab-bg);color:var(--lab-text);}
[data-testid="stHeader"]{background:var(--lab-bg);}
[data-testid="stSidebar"]{background:var(--lab-card);color:var(--lab-text);border-right:1px solid var(--lab-line);}
[data-testid="stMainBlockContainer"]{max-width:1320px;padding-top:3.5rem;padding-bottom:3rem;}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"], [data-testid="stMetric"]{color:var(--lab-text);}
[data-testid="stCaptionContainer"]{color:var(--lab-muted);}
h1,h2,h3{color:var(--lab-text);letter-spacing:-.035em;}
[data-testid="stExpander"],[data-testid="stAlert"]{background:var(--lab-card);color:var(--lab-text);border-color:var(--lab-line);}
[data-testid="stMetric"]{background:var(--lab-card);border:1px solid var(--lab-line);border-radius:16px;padding:20px;}
.lab-kicker{font-size:.75rem;letter-spacing:.16em;text-transform:uppercase;color:var(--lab-accent);font-weight:700;}
.lab-hero{padding:32px 0 12px;max-width:980px;}
.lab-hero h1{font-size:clamp(2.2rem,4.4vw,3.6rem);line-height:1.08;margin:16px 0 20px;max-width:850px;}
.lab-hero p{font-size:1.13rem;line-height:1.7;color:var(--lab-muted);max-width:740px;}
.lab-pills{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0;}
.lab-pill{border:1px solid var(--lab-line);background:var(--lab-card);border-radius:30px;padding:6px 12px;font-size:.8rem;color:var(--lab-text);}
.lab-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:16px 0 24px;}
.lab-card{background:var(--lab-card);border:1px solid var(--lab-line);border-radius:16px;padding:24px;}
.lab-card h3{font-size:1.12rem;margin:10px 0;}
.lab-card p{color:var(--lab-muted);font-size:.95rem;line-height:1.65;margin:8px 0;}
.lab-card .lab-number{color:var(--lab-accent);font-size:.8rem;font-weight:700;letter-spacing:.1em;}
.lab-profile{background:var(--lab-card);border:1px solid var(--lab-line);border-radius:18px;padding:28px;margin:24px 0;}
.lab-profile p,.lab-footer{color:var(--lab-muted);line-height:1.7;}
.lab-profile a,.lab-footer a{color:var(--lab-accent);}
a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid var(--lab-accent)!important;outline-offset:3px;}
.lab-footer{border-top:1px solid var(--lab-line);padding:24px 0;font-size:.85rem;}
@media(max-width:760px){.lab-grid{grid-template-columns:1fr;}.lab-hero{padding-top:8px;}[data-testid="stMainBlockContainer"]{padding-left:1rem;padding-right:1rem;}.lab-card{padding:20px;}}
@media(prefers-reduced-motion:reduce){.lab-hero,.lab-card{animation:none;transition:none;}}
</style>'''.replace('BG;', p['bg']+';').replace('CARD;', p['card']+';').replace('TEXT;', p['text']+';').replace('MUTED;', p['muted']+';').replace('LINE;', p['line']+';').replace('ACCENT;', p['accent']+';').replace('SOFT;', p['soft']+';'), unsafe_allow_html=True)


def cards(items):
    """Escape data before interpolating it into presentation HTML."""
    content = ''.join('<article class="lab-card"><span class="lab-number">'+escape(k)+'</span><h3>'+escape(title)+'</h3><p>'+escape(body)+'</p></article>' for k,title,body in items)
    st.markdown('<div class="lab-grid">'+content+'</div>', unsafe_allow_html=True)


def hero():
    st.markdown('''<section class="lab-hero"><div class="lab-kicker">RSSR LAB / Remote sensing × deep learning</div>
<h1>See more. Compare better.<br>Understand every pixel.</h1>
<p>A hands-on research laboratory for satellite image super-resolution. Explore seven CNN, attention and Transformer families, inspect measured results, and learn the mathematics behind reconstruction.</p>
<div class="lab-pills"><span class="lab-pill">Scientific reconstruction</span><span class="lab-pill">Seven model families</span><span class="lab-pill">Public guest access</span></div></section>''', unsafe_allow_html=True)


def overview_explorer(root, index, navigate):
    st.subheader('One laboratory. Three ways to explore.')
    cards([
        ('01 / EXPERIMENT', 'AI Studio', 'Start with a licensed sample or your own RGB image. Choose a checkpoint and scale, then inspect shared crops and export the actual output.'),
        ('02 / INVESTIGATE', 'Satellite Explorer', 'Define an area and acquisition dates. Inspect source metadata before reconstruction; display imagery is not calibrated reflectance.'),
        ('03 / UNDERSTAND', 'Learning Lab', 'Forty lessons across nine stages connect pixels, linear algebra, CNNs, attention and Transformers with bounded interactive experiments.'),
    ])
    for col,label,target in zip(st.columns(3), ['Open AI Studio','Explore a satellite region','Start learning'], ['Live SR','Satellite Explorer','Mini-projects']):
        col.button(label,key='overview-route-'+target,on_click=navigate,args=(target,),width='stretch')
    st.subheader('From input pixels to an inspectable result')
    cards([
        ('01–02', 'Choose & prepare', 'Upload an image or acquire a bounded satellite crop. Check dimensions, bands, source and declared numerical range.'),
        ('03–04', 'Configure & reconstruct', 'Choose a verified checkpoint and exact scale. Inspect the resource plan, then explicitly run native inference or a disclosed cascade.'),
        ('05–06', 'Compare & export', 'Compare at the same crop and contrast. Reference metrics require aligned ground truth. Download results with their provenance.'),
    ])
    from rssr.model_availability import availability
    families = availability(root)
    with st.expander('Inspect a recorded reconstruction figure'):
        figure = root/'reports/mp31-baseline-comparison.png'
        if figure.is_file():
            st.image(str(figure),caption='Recorded MP31 development comparison. Historical four-band validation; not today’s hosted RGB run or an independent satellite test.',width='stretch')
        else:
            st.info('The recorded figure is unavailable; run a comparison in AI Studio.')
    st.subheader('Seven architectures. Explicit evidence.')
    cards([(f"M{i} / "+('RGB AVAILABLE' if row['rgb_execution'] else 'UNAVAILABLE'), row['model'].split(' · ',1)[-1], row['status']) for i,row in enumerate(families,1)])
    st.caption('C/A/T identify additional feature branches. All families retain shared convolutional embedding/reconstruction. Availability is not a claim of geographic accuracy. Native ×2; larger configured scales use cascades.')
    st.button('Inspect model mathematics',key='overview-models',on_click=navigate,args=('Methodology & Data',))
    with st.expander('Browse the nine-stage learning roadmap'):
        for stage in dict.fromkeys(p['stage'] for p in index['projects']):
            projects = [p for p in index['projects'] if p['stage']==stage]
            st.markdown('**'+stage.replace('_',' ').title()+f' · {len(projects)} projects**')
            st.caption(' · '.join(p['id'] for p in projects))
        st.caption('Lesson availability, completed educational controls and independent scientific acceptance are different statuses. Inspect each lesson’s criteria.')


def researcher():
    st.markdown('''<section class="lab-profile"><div class="lab-kicker">About the researcher</div><h2>Ahmdullah</h2>
<p>Currently pursuing M.Tech in Signal Processing and Machine Learning<br>Indian Institute of Technology Guwahati</p>
<p>Research interests: deep learning, signal processing, computer vision, remote sensing and image super-resolution.</p>
<p><a href="mailto:exammotivation0816@gmail.com">Email Ahmdullah</a> · <a href="https://www.linkedin.com/in/exammotivation/">LinkedIn profile</a></p>
<p>Academic affiliation is researcher background; this project does not imply institutional endorsement.</p></section>''', unsafe_allow_html=True)


def footer():
    st.markdown('''<footer class="lab-footer"><strong>◈ RSSR LAB</strong> · Research, reconstruction and understanding.<br>
Ahmdullah · M.Tech student, SPML, IIT Guwahati · <a href="mailto:exammotivation0816@gmail.com">Contact</a> · <a href="https://www.linkedin.com/in/exammotivation/">LinkedIn</a><br>
Synthetic evidence ≠ verified ground truth. Generated texture is not verified geographic detail. Public results are temporary; no account or durable history is implied.</footer>''', unsafe_allow_html=True)
