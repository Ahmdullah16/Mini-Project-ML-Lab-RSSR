"""Shared presentation only: no inference, acquisition, account or training side effects."""
from html import escape
import streamlit as st

SECTION_LABELS = {
    'Overview': 'Overview', 'Live SR': 'AI Studio', 'Results': 'Results & benchmarks',
    'Methodology & Data': 'Model Lab & methodology', 'Mini-projects': 'Learning Lab',
    'Satellite Explorer': 'Satellite Explorer', 'Downloads & Reproducibility': 'Research & downloads',
}
from rssr_ui.tokens import PALETTES, css


def apply_theme(mode):
    # Style only our own surfaces; native controls retain supported Streamlit theming.
    reduced = st.session_state.get('aurora_reduce_motion', True) or st.query_params.get('lite') == '1'
    st.markdown(css(mode, reduced), unsafe_allow_html=True)


def cards(items):
    """Escape data before interpolating it into presentation HTML."""
    content = ''.join('<article class="lab-card"><span class="lab-number">'+escape(k)+'</span><h3>'+escape(title)+'</h3><p>'+escape(body)+'</p></article>' for k,title,body in items)
    st.markdown('<div class="lab-grid">'+content+'</div>', unsafe_allow_html=True)


def hero():
    st.markdown('''<section class="lab-hero"><div class="lab-kicker">RSSR AURORA X / EARTH INTELLIGENCE · IMAGE SUPER-RESOLUTION</div>
<h1>See more. Compare better.<br>Understand every pixel.</h1>
<p>A hands-on research laboratory for satellite image super-resolution. Explore seven CNN, attention and Transformer families, inspect measured results, and learn the mathematics behind reconstruction.</p>
<div class="lab-pills"><span class="lab-pill">Scientific reconstruction</span><span class="lab-pill">Seven model families</span><span class="lab-pill">Public guest access</span></div></section>''', unsafe_allow_html=True)


def overview_explorer(root, index, navigate):
    st.subheader('One laboratory. Three ways to explore.')
    cards([
        ('01 / EXPERIMENT', 'AI Studio', 'Start with a licensed sample or your own RGB image. Choose a checkpoint and scale, then inspect shared crops and export the actual output.'),
        ('02 / INVESTIGATE', 'Satellite Explorer', 'Define an area and acquisition dates. Inspect source metadata before reconstruction; display imagery is not calibrated reflectance.'),
        ('03 / UNDERSTAND', 'Learning Lab', f"{len(index['projects'])} lessons across {len({p['stage'] for p in index['projects']})} stages connect pixels, linear algebra, CNNs, attention and Transformers with bounded interactive experiments."),
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
    st.markdown('''<footer class="lab-footer"><strong>◈ RSSR AURORA X</strong> · Research, reconstruction and understanding.<br>
Ahmdullah · M.Tech student, SPML, IIT Guwahati · <a href="mailto:exammotivation0816@gmail.com">Contact</a> · <a href="https://www.linkedin.com/in/exammotivation/">LinkedIn</a><br>
Synthetic evidence ≠ verified ground truth. Generated texture is not verified geographic detail. Public results are temporary; no account or durable history is implied.</footer>''', unsafe_allow_html=True)
