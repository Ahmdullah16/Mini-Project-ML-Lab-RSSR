"""Evidence-first showcase. No model loading, network or training on navigation."""
import json
from pathlib import Path
import sys
import pandas as pd
import altair as alt
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'app'))
from rssr.evidence import build_index
from rssr.build_info import build_identity
REPO='https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR'
SECTIONS=['Overview','Live SR','Results','Methodology & Data','Mini-projects','Satellite Explorer','Downloads & Reproducibility']
st.set_page_config(page_title='RSSR · Research laboratory',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container{max-width:1240px;padding-top:4rem;padding-bottom:3rem;}h1{letter-spacing:-.035em;}h2{letter-spacing:-.02em;}
[data-testid="stMetric"]{background:rgba(37,99,128,.065);border:1px solid rgba(37,99,128,.2);border-radius:12px;padding:16px;}
[data-testid="stSidebar"]{border-right:1px solid rgba(37,99,128,.18);}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem;}h1{font-size:2rem;}}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def evidence(fingerprint):
    return build_index(ROOT)

evidence_paths=[*(ROOT/'reports').glob('*'),ROOT/'docs/project_folders.json',ROOT/'docs/syllabus.json',*ROOT.glob('stage_*/MP*/README.md')]
fingerprint=tuple((p.relative_to(ROOT).as_posix(),p.stat().st_mtime_ns) for p in sorted(evidence_paths) if p.is_file())
try:index=evidence(fingerprint)
except (OSError,ValueError,KeyError) as error:
    st.error(f'Evidence could not be loaded: {error}. No results have been substituted.');st.stop()
with st.sidebar:
    st.markdown('### ◈ RSSR LAB');st.caption('REMOTE SENSING · SUPER-RESOLUTION')
    section=st.radio('Explore the laboratory',SECTIONS,key='section',label_visibility='collapsed')
    st.divider();st.caption('Measured evidence. Reproducible experiments. Explicit limits.')
    st.link_button('Source & notebooks',REPO)
    with st.expander('Application version'):st.json(build_identity(ROOT))
st.caption('RSSR / '+section.upper())

def download_artifact(path):
    file=ROOT/path
    if file.is_file():st.download_button('Download '+file.name,file.read_bytes(),file.name,key='download-'+path)
    else:st.caption(path+' — not yet available; not marked complete.')

def metric_chart(frame, title):
    chart = alt.Chart(frame).mark_bar(color='#167d9a').encode(
        x=alt.X('value:Q', title=title), y=alt.Y('model:N', title='Model'),
        tooltip=['model:N', 'value:Q'])
    st.altair_chart(chart, width='stretch')

if section=='Overview':
    st.title('More pixels. Measured evidence.')
    st.markdown('Explore how convolution, attention and windowed Transformers reconstruct detail in remote-sensing imagery—and where the evidence stops.')
    a,b,c=st.columns(3)
    a.metric('Frozen test scenes','1',help='One held-out NAIP aerial scene; 16 fixed patches. Not broad geographic validation.')
    b.metric('Development run records','36',help='Includes six reused runs; not 36 new training jobs.')
    expanded=sum(not p['status'].startswith('Introductory') for p in index['projects'])
    c.metric('Expanded lessons',f'{expanded} / 40',help='See each project for criterion-level completion; learner mastery remains unassessed.')
    st.subheader('What the experiments support')
    test=pd.DataFrame([r for r in index['rows'] if r['study']=='Frozen held-out evaluation' and r['bands']==3 and r['metric']=='psnr'])
    metric_chart(test, 'PSNR · dB')
    st.caption('RGB · test · synthetic NAIP x2 · one scene / 16 patches · range 1, border 2. Official SwinIR has different pretraining. Higher PSNR is better; bars are not uncertainty intervals.')
    st.info('CNN and hybrid are effectively tied in this small experiment. More elaborate architecture does not establish superiority or novelty.')
    left,right=st.columns(2)
    with left:
        st.subheader('The approach');st.write('A local CNN captures nearby structure. Feature attention and an unshifted window Transformer provide complementary context. A learned residual corrects bicubic interpolation.')
        st.caption('Compact educational architecture; not an exact RCAN, CBAM or SwinIR reproduction.')
    with right:
        st.subheader('Completion, without shortcuts')
        st.markdown(f'**Engineering:** public hybrid inference recorded; stateless service.\n\n**Education:** {expanded} lessons expanded; inspect project-specific completion criteria.\n\n**Research:** frozen synthetic test executed; paired cross-sensor validity unestablished.\n\n**Submission:** final report, slides and complete package pending.')
    with st.expander('Why remote sensing needs more than sharper previews'):
        st.write('Generated texture can look convincing without matching the ground. Sensor response, registration, sampling and physical calibration matter. Display previews and synthetic PSNR do not establish reflectance or downstream accuracy.')
elif section=='Live SR':
    st.title('Inspect a reconstruction');st.caption('Choose a genuine checkpoint and compare a shared crop. Uploads without references receive no invented accuracy metrics.')
    from inference_ui import render_live
    render_live()
elif section=='Results':
    st.title('The evidence, including the misses')
    st.caption('Each view retains its study, split, bands and protocol. Historical pilots and teaching reconstruction are not frozen test results.')
    view=st.radio('Evidence view',['Measured tables','Learning curves','Artifact library'],horizontal=True,key='result_view')
    if view=='Measured tables':
        frame=pd.DataFrame(index['rows']);studies=list(dict.fromkeys(frame.study))
        study=st.selectbox('Study',studies,index=studies.index('Frozen held-out evaluation'));frame=frame[frame.study==study]
        a,b=st.columns(2);bands=a.selectbox('Bands',sorted(frame.bands.unique()),format_func=lambda n:'RGB' if n==3 else 'RGB + NIR');frame=frame[frame.bands==bands]
        metric=b.selectbox('Metric',list(dict.fromkeys(frame.metric)));frame=frame[frame.metric==metric]
        condition=st.selectbox('Condition',list(dict.fromkeys(frame.condition)));frame=frame[frame.condition==condition]
        query=st.text_input('Search model or source run',placeholder='hybrid, bicubic, run ID…')
        if query:frame=frame[frame.model.str.contains(query,case=False,regex=False)|frame.run_id.fillna('').str.contains(query,case=False,regex=False)]
        if frame.empty:st.info('No recorded values match. Clear the search to restore the table.')
        else:
            metric_chart(frame.groupby('model',sort=False).value.mean().reset_index(), metric+' · '+frame.iloc[0]['unit'])
            st.caption(f"{frame.iloc[0]['unit']} · {frame.iloc[0]['split']} · {frame.iloc[0]['task']} · {frame.iloc[0]['dataset']}. Chart averages displayed rows; the table retains every source record. Do not pool different studies.")
            st.dataframe(frame[['model','value','unit','split','scenes','tiles','seed','run_id','condition','protocol','source']],hide_index=True,width='stretch')
            st.download_button('Download filtered CSV',frame.to_csv(index=False),'rssr-filtered-results.csv','text/csv')
        if study=='CPU efficiency':st.warning('Sampled process RSS is not allocator peak. Partial MACs omit normalization, activation, pooling and interpolation. Latency is hardware-specific.')
        if study=='Robustness / spectral':st.warning('Spectral metrics use normalized DN, not reflectance. Synthetic blur/noise is not cross-sensor validation.')
        if study=='Official pretrained baseline':st.info('Official SwinIR uses DIV2K natural RGB and bicubic pretraining. Budgets and domains differ.')
        with st.expander('Metric and source conventions'):
            st.write(index['limits']);st.write('Higher PSNR/SSIM is better; lower SAM/RMSE is better. Missing values are unavailable, never zero. Repeated bicubic records retain seed/source context; they are not independent scenes.')
    elif view=='Learning curves':
        curves=pd.DataFrame(index['curves']);metric=st.selectbox('Recorded loss',['validation_l1','train_l1'])
        curves['series']=curves.model+' · seed '+curves.seed.astype(str)
        st.line_chart(curves.pivot(index='step',columns='series',values=metric))
        st.caption('Recorded MP26 development: 1000 steps, Adam .001, batch 4, seeds 17/23/41. Training points are minibatch losses; validation selects checkpoints. No test tuning.')
    else:
        query=st.text_input('Search recorded artifacts',placeholder='ablation, pilot, mp…')
        artifacts=[a for a in index['artifacts'] if query.lower() in a['path'].lower()]
        st.dataframe(pd.DataFrame(artifacts),hide_index=True,width='stretch')
        if artifacts:
            chosen=st.selectbox('Inspect or download source',[a['path'] for a in artifacts])
            if chosen.endswith('.png'):st.image(str(ROOT/chosen),width='stretch')
            download_artifact(chosen)
        else:st.info('No artifacts match.')
elif section=='Methodology & Data':
    st.title('From geographic split to frozen test')
    st.markdown('**Source scenes → geographic split → valid patches → synthetic LR → training / validation → frozen selection → held-out evaluation.**')
    st.graphviz_chart('digraph {rankdir=LR; node [shape=box,style=rounded]; LR -> Head; Head -> "CNN + attention"; Head -> "Window Transformer"; "CNN + attention" -> Fusion; "Window Transformer" -> Fusion; Fusion -> PixelShuffle; LR -> Bicubic; Bicubic -> Add; PixelShuffle -> Add; Add -> SR;}')
    st.latex(r'\hat{x}=B(y)+R_\theta(y),\qquad \mathcal{L}=\frac{1}{N}\sum_i|\hat{x}_i-x_i|')
    st.write('N×C×H×W → width-16 features → C×4 tail channels → N×C×2H×2W. C=3 RGB or4 RGBNIR. The hosted display path supports RGB.')
    source=json.loads((ROOT/'configs/naip-rgb-development-manifest.json').read_text())
    st.subheader('Data provenance and geography')
    st.dataframe(pd.DataFrame([{k:s.get(k) for k in ['id','region_id','split','acquisition_time','sha256']} for s in source['scenes']]),hide_index=True,width='stretch')
    st.caption(source['license']+' · '+source['units'])
    st.write('Four train, four validation, one held-out scene. Sixteen nonoverlapping 32×32 patches per scene in deterministic raster order; not representative random sampling. Split before tiling; reject geographic overlap and a0.02-degree buffer violation. This angular buffer is not a fixed ground distance.')
    st.subheader('Comparable metrics, bounded conclusions')
    st.write('Gaussian σ=.8 HR pixels, area downsampling x2. Float32, fixed DN/10000, range1, border2. No clipping or luminance conversion before evaluation. SSIM:7×7 population moments. Tile metrics aggregate within scenes. Three seeds measure training variation; four validation scenes provide limited geographic evidence.')
    with st.expander('Frozen protocol and checkpoint provenance'):st.json(json.loads((ROOT/'configs/final-evaluation.json').read_text()))
    st.warning('Reflectance calibration, real cross-sensor alignment and downstream benefit remain unvalidated. Page navigation never repeats the frozen test.')
elif section=='Mini-projects':
    st.title('Forty projects. Nine connected stages.')
    st.caption('Expanded and executed does not imply every criterion satisfied or mastery assessed. MP20 retains a missing independent SRCNN held-out table.')
    query=st.text_input('Find a project',placeholder='MP26, attention, metrics…');found=False
    for stage in dict.fromkeys(p['stage'] for p in index['projects']):
        matches=[p for p in index['projects'] if p['stage']==stage and query.lower() in (p['id']+' '+p['title']+' '+p['objective']).lower()]
        if not matches:continue
        found=True;st.subheader(stage.replace('_',' ').title())
        for p in matches:
            with st.expander(p['id']+' · '+p['title']):
                st.write(p['objective']);st.caption(p['status'])
                if p['exception']:st.warning(p['exception'])
                st.markdown(f"[Project brief]({REPO}/blob/main/{p['brief']}) · [Notebook]({REPO}/blob/main/{p['notebook']})")
                figures=[a['path'] for a in index['artifacts'] if a['path'].lower().startswith('reports/'+p['id'].lower()+'-') and a['path'].endswith('.png')]
                if figures:st.image(str(ROOT/figures[0]),width='stretch')
    if not found:st.info('No projects match. Clear the search to see all40.')
elif section=='Satellite Explorer':
    st.title('Acquire a bounded satellite crop');st.caption('Explicit acquisition, not an automatic live feed. No provider request until you submit; optional storage credentials are not required.')
    from inference_ui import render_satellite
    render_satellite()
else:
    st.title('Take the evidence with you')
    st.caption('Measured records and reproducibility instructions. Unfinished artifacts are labeled pending, not replaced by placeholders.')
    st.download_button('Download results index JSON',json.dumps(index,indent=2,allow_nan=False),'rssr-evidence-index.json','application/json')
    st.download_button('Download all indexed metrics CSV',pd.DataFrame(index['rows']).to_csv(index=False),'rssr-metrics.csv','text/csv')
    st.download_button('Download artifact manifest',json.dumps(index['artifacts'],indent=2),'rssr-artifacts.json','application/json')
    for path in ['docs/examiner-demo.md','docs/frozen-reproduction.md','docs/setup.md','references.bib','docs/submission_readiness.md']:download_artifact(path)
    st.warning('Pending: '+', '.join(index['pending_artifacts'])+'. Recorded research drafts and figures remain in Results → Artifact library.')
    st.code('python -m pip install -r app/requirements.txt\npython scripts/check_model_catalog.py\npython -m streamlit run app/streamlit_app.py',language='bash')
    st.write('Follow the research setup guide for dataset rebuilding. Original checkpoint and tile hashes remain authoritative. Public app inputs, results and caches are temporary; live external durability needs a separately configured and verified account.')
st.divider();st.caption('RSSR LAB · Synthetic evidence ≠ verified ground truth · No training on navigation')
