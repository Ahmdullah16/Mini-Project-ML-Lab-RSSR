"""Native explanatory content; no notebook viewer, execution, or network."""
import streamlit as st
import pandas as pd
from rssr.learning_content import load_lesson, visitor_prose


def render(root, projects):
    st.markdown('<span id="shared-concepts"></span>', unsafe_allow_html=True)
    with st.expander('Shared reference · dimensions, losses and scientific interpretation'):
        st.markdown((root/'docs/learning-reference.md').read_text(encoding='utf-8'))
        width=st.slider('Teaching input width (pixels)',1,1024,256,key='teaching_width')
        height=st.slider('Teaching input height (pixels)',1,1024,128,key='teaching_height')
        scale=st.selectbox('Teaching scale',[2,4,8,16,32,64],key='teaching_scale')
        st.write(f'{width} × {height} → {width*scale} × {height*scale}; {width*height*scale*scale:,} output pixels.')
        st.line_chart(pd.DataFrame({'Output pixels':[width*height*s*s for s in [2,4,8,16,32,64]]},index=[2,4,8,16,32,64]),x_label='Scale',y_label='Pixels')
        st.caption('Arithmetic teaching example only. This control does not run inference or imply execution availability.')
    if not projects:
        return
    ident=st.selectbox('Read a lesson here',[p['id'] for p in projects],key='onsite_lesson')
    lesson=load_lesson(root,ident)
    st.caption(lesson['scope'])
    st.info(lesson['coverage'])
    st.markdown('[Shared mathematical reference](#shared-concepts)')
    st.markdown(visitor_prose(lesson['overview']))
    if ident=='MP01':
        from pixel_lab_ui import render as render_pixel_lab
        render_pixel_lab(root)
    elif ident=='MP02':
        from array_lab_ui import render as render_array_lab
        render_array_lab()
    elif ident=='MP03':
        from resize_lab_ui import render as render_resize_lab
        render_resize_lab()
    elif ident=='MP04':
        from difference_lab_ui import render as render_difference_lab
        render_difference_lab()
    elif ident=='MP05':
        from blur_lab_ui import render as render_blur_lab
        render_blur_lab()
    elif ident in ('MP06','MP07','MP08'):
        from degradation_batch_ui import render as render_degradation_lab
        render_degradation_lab(ident)
    elif ident in ('MP09','MP10','MP11'):
        from metric_lesson_ui import render as render_metric_lesson
        render_metric_lesson(ident)
    elif ident in ('MP12','MP13','MP14','MP15','MP16'):
        from math_training_ui import render as render_math_training
        render_math_training(ident)
    elif ident=='MP17':
        from convolution_lesson_ui import render as render_convolution
        render_convolution()
    elif ident in ('MP18','MP19','MP20','MP21','MP22'):
        from cnn_native_lessons import render as render_cnn
        render_cnn(ident)
    with st.expander('Derivations, worked examples and interpretation',expanded=True):
        st.markdown(visitor_prose(lesson['explanations']))
    with st.expander('Recorded execution and limitations'):
        st.markdown(visitor_prose(lesson['validation']))
