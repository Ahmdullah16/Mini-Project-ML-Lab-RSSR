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
from rssr.interpretation import explain_selection, metric_definition
from rssr.build_info import build_identity
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

def navigate(section):
    st.session_state['section']=section

def explain_results(frame):
    explanation=explain_selection(frame.to_dict('records'))
    if not explanation['available']:
        st.info(explanation.get('message','Values unavailable.'));return
    st.markdown('**Question:** '+explanation['question'])
    st.write(explanation['explanation'])
    st.caption('Direction: '+explanation['direction']+'. Horizontal axis: '+explanation['context']['unit']+'; vertical axis: model. Blue bars identify recorded means, not confidence or quality categories.')
    st.markdown('**Observed in this selection:** '+explanation['observation'])
    with st.expander('Inputs, aggregation, source records and limits'):
        st.write(explanation['context'])
        st.write(explanation['aggregation']);st.write(explanation['limits'])
        for source in explanation['sources']:
            st.caption('Recorded evidence: '+source)
    with st.expander('Metric mathematics and a controlled numerical example'):
        st.latex(r'\mathrm{MSE}=\frac{1}{N}\sum_i(\hat{x}_i-x_i)^2,\quad \mathrm{RMSE}=\sqrt{\mathrm{MSE}},\quad \mathrm{MAE}=\frac{1}{N}\sum_i|\hat{x}_i-x_i|')
        st.latex(r'\mathrm{PSNR}=10\log_{10}(L^2/\mathrm{MSE})')
        st.write('N counts valid scalar samples, including channels. L is the declared range, not an image maximum. Exact equality gives zero error and infinite PSNR; an empty valid mask is undefined. Nonfinite inputs must be rejected or masked before computation.')
        st.latex(r'\mathrm{SSIM}=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}')
        st.write('μ denotes local mean, σ² population variance and σxy population covariance. Shared evaluator: uniform 7×7 fully valid windows, C1=(0.01L)², C2=(0.03L)², then average over windows and channels. An undersized/no-valid-window result is unavailable.')
        st.latex(r'\mathrm{SAM}=\arccos\!\left(\frac{x\cdot\hat{x}}{\|x\|\,\|\hat{x}\|}\right)')
        st.write('SAM averages radians over valid spectral vectors; zero-norm products ≤1e-12 are excluded. Cosines are clipped to [-1,1]. RGB display values do not establish calibrated multispectral reflectance.')
        st.latex(r'\Delta\mathrm{PSNR}=\mathrm{PSNR}_{model}-\mathrm{PSNR}_{bicubic},\quad \mathrm{RMSE\ reduction}=100\frac{\mathrm{RMSE}_{bicubic}-\mathrm{RMSE}_{model}}{\mathrm{RMSE}_{bicubic}}')
        st.write('RMSE percentage is unavailable when the baseline denominator is zero. A percentage of a dB value is not a meaningful quality improvement. Mean PSNR cannot be inverted into mean MSE.')
        st.write('Controlled teaching example (not a research observation): x=[0.1,0.2], prediction=[0.2,0.2], two valid scalar samples, L=1. MSE=(0.1²+0²)/2=0.005; RMSE≈0.07071; MAE=0.05; PSNR≈23.0103 dB. This calculation requires a genuine aligned reference when used on images.')

if section=='Overview':
    st.title('More pixels. Measured evidence.')
    st.markdown('Explore how convolution, attention and windowed Transformers reconstruct detail in remote-sensing imagery—and where the evidence stops.')
    st.write('Research question: under the same synthetic degradation and validation budget, does combining CNN, attention and Transformer features improve reconstruction over simpler baselines?')
    for col,label,target in zip(st.columns(3),['Try inference','Explore results','Read methodology'],['Live SR','Results','Methodology & Data']):
        col.button(label,on_click=navigate,args=(target,),width='stretch')
    a,b,c=st.columns(3)
    frozen_rows=[r for r in index['rows'] if r['study']=='Frozen held-out evaluation']
    a.metric('Frozen test scenes',max(r['scenes'] for r in frozen_rows),help='Count from frozen source records; not summed across models.')
    b.metric('Development run records',len({r['run_id'] for r in index['rows'] if r['study']=='Development seed summaries'}),help='Distinct source run IDs; includes reused runs, not all new training jobs.')
    expanded=sum(not p['status'].startswith('Introductory') for p in index['projects'])
    c.metric('Expanded lessons',f'{expanded} / 40',help='See each project for criterion-level completion; learner mastery remains unassessed.')
    st.subheader('What the experiments support')
    test=pd.DataFrame([r for r in index['rows'] if r['study']=='Frozen held-out evaluation' and r['bands']==3 and r['metric']=='psnr'])
    metric_chart(test, 'PSNR · dB')
    st.caption('RGB · test · synthetic NAIP x2 · one scene / 16 patches · range 1, border 2. Official SwinIR has different pretraining. Higher PSNR is better; bars are not uncertainty intervals.')
    explain_results(test)
    st.info('CNN and hybrid are effectively tied in this small experiment. More elaborate architecture does not establish superiority or novelty.')
    left,right=st.columns(2)
    with left:
        st.subheader('The approach');st.write('A local CNN captures nearby structure. Feature attention and an unshifted window Transformer provide complementary context. A learned residual corrects bicubic interpolation.')
        st.caption('Compact educational architecture; not an exact RCAN, CBAM or SwinIR reproduction.')
    with right:
        st.subheader('Completion, without shortcuts')
        st.markdown(f'**Engineering:** historical public hybrid inference recorded; current public access unverified. Stateless service.\n\n**Education:** {expanded} lessons expanded; inspect project-specific completion criteria.\n\n**Research:** frozen synthetic test executed; paired cross-sensor validity unestablished.\n\n**Submission:** final report, slides and complete package pending.')
    with st.expander('Why remote sensing needs more than sharper previews'):
        st.write('Generated texture can look convincing without matching the ground. Sensor response, registration, sampling and physical calibration matter. Display previews and synthetic PSNR do not establish reflectance or downstream accuracy.')
elif section=='Live SR':
    st.title('Inspect a reconstruction');st.caption('Choose a genuine checkpoint and compare a shared crop. Uploads without references receive no invented accuracy metrics.')
    from metric_ui import render as render_metrics
    render_metrics('live')
    from availability_ui import render as render_availability
    render_availability()
    workflow=st.radio('Inference workflow',['Single model','Compare models'],horizontal=True,key='inference_workflow')
    if workflow=='Single model':
        from inference_ui import render_live
        render_live()
    else:
        from compare_ui import render_compare
        render_compare()
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
        task=st.selectbox('Task',list(dict.fromkeys(frame.task)));frame=frame[frame.task==task]
        protocol=st.selectbox('Protocol',list(dict.fromkeys(frame.protocol)));frame=frame[frame.protocol==protocol]
        query=st.text_input('Search model or source run',placeholder='hybrid, bicubic, run ID…')
        if query:frame=frame[frame.model.str.contains(query,case=False,regex=False)|frame.run_id.fillna('').str.contains(query,case=False,regex=False)]
        if frame.empty:st.info('No recorded values match. Clear the search to restore the table.')
        else:
            explain_results(frame)
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
        st.write(metric_definition(metric)['explanation'])
        st.caption('Question: how does the recorded loss change over optimizer steps? Horizontal axis: step; vertical axis: normalized-DN absolute error. Each colored line is one model/seed, not a confidence interval. Compare training and validation trends before claiming overfitting; these views alone do not establish its cause.')
        st.caption('Curve evidence: recorded MP26 experiment')
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
    from model_explorer_ui import render as render_models
    render_models(ROOT)
    with st.expander('Contribution audit · what the evidence supports'):
        audit=json.loads((ROOT/'docs/contribution-audit.json').read_text(encoding='utf-8'))
        st.caption(audit['scope'])
        for item in audit['items']:
            st.subheader(item['proposed'])
            for field in ['prior_work','difference','needed','observed','supported_claim']:
                st.markdown('**'+field.replace('_',' ').title()+':** '+item[field])
            st.markdown(' · '.join(f'[Primary source {i+1}]({url})' for i,url in enumerate(item['references'])))
    st.divider()
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
    from learning_ui import render as render_learning
    render_learning(ROOT,[p for p in index['projects'] if query.lower() in (p['id']+' '+p['title']+' '+p['objective']).lower()])
    for stage in dict.fromkeys(p['stage'] for p in index['projects']):
        matches=[p for p in index['projects'] if p['stage']==stage and query.lower() in (p['id']+' '+p['title']+' '+p['objective']).lower()]
        if not matches:continue
        found=True;st.subheader(stage.replace('_',' ').title())
        for p in matches:
            with st.expander(p['id']+' · '+p['title']):
                st.write(p['objective']);st.caption(p['status'])
                if p['exception']:st.warning(p['exception'])
                st.caption("Read this project in the on-site lesson selector above.")
                figures=[a['path'] for a in index['artifacts'] if any(part.lower().startswith(p['id'].lower()+'-') for part in Path(a['path']).parts) and Path(a['path']).suffix.lower() in ('.png','.svg','.jpg','.jpeg')]
                for figure in figures:
                    st.image(str(ROOT/figure),caption=Path(figure).name,width='stretch')
    if not found:st.info('No projects match. Clear the search to see all40.')
elif section=='Satellite Explorer':
    st.title('Acquire a bounded satellite crop');st.caption('Explicit acquisition, not an automatic live feed. No provider request until you submit; optional storage credentials are not required.')
    st.info('Current acquisition is display-only Sentinel L2A visual uint8 RGB with item-level cloud filtering. It is not calibrated surface reflectance. The new local scientific conversion layer is tested on fixtures; provider acquisition/alignment and scientific model validation are still pending.')
    from availability_ui import render as render_availability
    render_availability()
    from inference_ui import render_satellite
    render_satellite()
    from metric_ui import render as render_metrics
    render_metrics("satellite")
else:
    st.title('Take the evidence with you')
    st.caption('Measured records and reproducibility instructions. Unfinished artifacts are labeled pending, not replaced by placeholders.')
    st.download_button('Download results index JSON',json.dumps(index,indent=2,allow_nan=False),'rssr-evidence-index.json','application/json')
    st.download_button('Download all indexed metrics CSV',pd.DataFrame(index['rows']).to_csv(index=False),'rssr-metrics.csv','text/csv')
    st.download_button('Download artifact manifest',json.dumps(index['artifacts'],indent=2),'rssr-artifacts.json','application/json')
    for path in ['paper/supplement.md','reports/learning-content-manifest.json','docs/contribution-audit.json','docs/examiner-demo.md','docs/frozen-reproduction.md','docs/setup.md','references.bib','docs/submission_readiness.md']:download_artifact(path)
    st.warning('Pending: '+', '.join(index['pending_artifacts'])+'. Recorded research drafts and figures remain in Results → Artifact library.')
    st.code('python -m pip install -r app/requirements.txt\npython scripts/check_model_catalog.py\npython -m streamlit run app/streamlit_app.py',language='bash')
    st.write('Follow the research setup guide for dataset rebuilding. Original checkpoint and tile hashes remain authoritative. Public app inputs, results and caches are temporary; live external durability needs a separately configured and verified account.')
st.divider();st.caption('RSSR LAB · Synthetic evidence ≠ verified ground truth · No training on navigation')
