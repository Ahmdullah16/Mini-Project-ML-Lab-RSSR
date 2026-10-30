"""Native model cards from recorded evidence; no model import or inference."""
import json
import streamlit as st

NAMES={'M1':'CNN','M2':'Attention','M3':'Transformer','M4':'CNN + Attention',
       'M5':'CNN + Transformer','M6':'Attention + Transformer','M7':'CNN + Attention + Transformer'}
PATHS={'M1':'Embedding → CNN → Reconstruction','M2':'Embedding → Feature attention → Reconstruction',
       'M3':'Embedding → Transformer → Reconstruction','M4':'Embedding → CNN → Feature attention → Reconstruction',
       'M5':'Embedding → parallel CNN / Transformer → Concat projection → Reconstruction',
       'M6':'Embedding → Transformer → Feature attention → Reconstruction',
       'M7':'Embedding → parallel CNN / Transformer → Concat projection → Feature attention → Reconstruction'}


def render(root):
    st.subheader('Model Explorer · seven distinct configurations')
    st.caption('Scientific reconstruction · educational cat-v1 family. Advanced configurations and their ablations are not yet implemented or validated.')
    path=root/'reports/historical-architecture-bindings.json'
    if not path.exists():
        st.warning('Historical checkpoint evidence is unavailable. No trained status has been substituted.');return
    records=json.loads(path.read_text(encoding='utf-8'))['rows']
    pilot_path=root/'reports/missing-model-pilot.json'
    pilots=json.loads(pilot_path.read_text(encoding='utf-8'))['runs'] if pilot_path.exists() else []
    extra=root/'reports/rgb-compatible-pilot.json'
    if extra.exists():pilots+=json.loads(extra.read_text(encoding='utf-8'))['runs']
    for cols,start in [(st.columns(3),0),(st.columns(3),3),(st.columns(1),6)]:
        for col,key in zip(cols,list(NAMES)[start:start+len(cols)]):
            bindings=[r for r in records if r['status']=='verified historical binding' and r['registry_config']['architecture_id']=='cat-v1/'+key]
            pilot_runs=[r for r in pilots if r['architecture_id']=='cat-v1/'+key]
            with col.container(border=True):
                st.markdown(f'**{key} · {NAMES[key]}**')
                st.caption('Implemented · '+(f'{len(bindings)} historical pilot bindings' if bindings else f'{len(pilot_runs)} bounded training pilots' if pilot_runs else 'Training evidence unavailable'))
                st.write(PATHS[key])
                if bindings:
                    st.write('Recorded native scales: '+', '.join('×'+str(x) for x in sorted({r['registry_config']['scale'] for r in bindings})))
                    st.caption('Binding is specific to configuration and checkpoint; it does not establish deployment or geographic validity.')
                elif pilot_runs:st.caption('Native ×2 RGB only. Pinned release assets verified for RGB execution; research acceptance remains pending.')
                else:st.caption('No trained output offered for this configuration.')
    key=st.selectbox('Inspect architecture',list(NAMES),format_func=lambda k:k+' · '+NAMES[k],key='registry_architecture')
    st.write(PATHS[key]+' + bicubic input baseline')
    from architecture_diagrams import diagram
    st.graphviz_chart(diagram(key))
    with st.expander('Derive the implemented operations and parameter effects',expanded=True):
        st.markdown((root/'docs/model-instructor.md').read_text(encoding='utf-8'))
    st.latex(r'\widehat{x}=B(y)+R(g(E(y)))')
    st.write('y is the input; E embeds bands; g is the selected feature path; R reconstructs output pixels; B is bicubic interpolation. C denotes the additional convolutional backbone, A additional channel/spatial attention, and T a Transformer including self-attention. Every model retains shared embedding/reconstruction convolutions.')
    rows=[r for r in records if r['status']=='verified historical binding' and r['registry_config']['architecture_id']=='cat-v1/'+key]
    if rows:
        chosen=st.selectbox('Recorded checkpoint',rows,format_func=lambda r:f"{r['registry_config']['bands']} bands · width {r['registry_config']['width']} · seed {r['full_training_config']['seed']} · {r['run_id'][:8]}",key='registry_checkpoint_'+key)
        st.write(f"Parameters: {chosen['parameters']:,} · selected optimizer step: {chosen['selected_step']} · native ×{chosen['registry_config']['scale']}")
        st.caption('Checkpoint SHA256: '+chosen['checkpoint_sha256'])
    selected_pilots=[r for r in pilots if r['architecture_id']=='cat-v1/'+key]
    if selected_pilots:
        chosen=st.selectbox('Bounded pilot checkpoint',selected_pilots,format_func=lambda r:f"Seed {r['seed']} · native ×2 · step {r['selected_step']}",key='registry_pilot_checkpoint_'+key)
        st.write(f"Parameters: {chosen['parameters']:,} · training: {chosen['seconds_training']:.3f} seconds")
        st.caption('Checkpoint SHA256: '+chosen['checkpoint_sha256'])
        st.warning('Development-only pilot: 100 Adam steps, width 16, minimum validation L1 selection. No test evaluation or matched-budget claim. Pinned deployment availability is listed in Live SR and Satellite Explorer. M2 budget matching remains unsuccessful.')
        st.dataframe(chosen['validation_scene_means'],hide_index=True)
        st.caption('PSNR is averaged over tiles separately within each validation scene; these are reused development scenes, not independent test evidence. Higher PSNR means lower squared error for the same range and border convention.')
    st.markdown('**What this shows:** distinct feature paths and verified historical artifact compatibility. **What it cannot show:** that more modules improve reconstruction, that all scales are trained natively, or that the models transfer to satellite reflectance.')
    st.info('At ×64, pixel count increases by 4096. Nominal output spacing equals input spacing / 64; validated resolving power remains unavailable without appropriate experiments.')
    with st.expander('Optional reproducibility evidence'):
        st.download_button('Download checkpoint binding evidence',path.read_bytes(),path.name,'application/json',key='registry_evidence')
