"""MP32–36 native synthetic lessons, distinct from real-data scientific acceptance."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from remote_sensing_labs import ROOT, raster, normalization, pairing, geography, feasibility


def table(label, values):
    st.write(label)
    st.dataframe(pd.DataFrame(values))


def bands(label, values):
    for i, band in enumerate(values):
        table(f'{label} · band {i} (rows × columns)', band)


def render(ident,root=ROOT):
    torch.set_num_threads(2)
    st.subheader(ident+' · native remote-sensing laboratory')
    st.info('Synthetic teaching fixtures only. No imagery acquisition, frozen-test access, training or checkpoint changes. '
            'Arrays and masks remain separate; numerical band values are not a calibrated reflectance or display RGB product.')
    settings = {'project':ident,'device':'CPU','scope':'synthetic teaching, not measured sensor/SR performance'}
    if ident == 'MP32':
        mode = st.selectbox('Validity mask', ['exclude sentinel','include all','all invalid'],key='MP32_mask')
        settings['mask'] = mode
        st.latex(r'(x,y)=T(c+1/2,r+1/2),\quad \mu_b=\frac{\sum_{r,c}m_{b,r,c}x_{b,r,c}}{\sum_{r,c}m_{b,r,c}}')
        st.write('T is the full affine map; r=row, c=column, b=band and m=validity. The synthetic north-up grid has '
                 '10 m pixels and upper-left corner (100,200). Row2,column3 center is (135,175); its corner is (130,180). '
                 'This coordinate example is separate from the 2×2 statistics fixture. Excluding its lower-left sentinel99 '
                 'leaves band0 values [1,3,5], mean3, and band1 [2,4,6], mean4. Including99 raises band0 mean to27. '
                 'An all-invalid band has unavailable statistics, not mean zero; valid zero is still an observation.')
        st.graphviz_chart('digraph {rankdir=LR; "CHW samples" -> "Finite values AND per-band mask" -> "Valid statistics"; "Row / column" -> Affine -> "Map coordinates + CRS";}')
        st.caption('A nodata mask does not identify every cloud. CRS units, band semantics, scale/offset and acquisition time need source metadata; no real source is loaded here.')
        compute = lambda:raster(mode)
    elif ident == 'MP33':
        split = st.selectbox('Attempt to fit statistics on', ['train','validation'],key='MP33_split')
        threshold = st.selectbox('Minimum common-valid patch fraction', [1.,.75],key='MP33_threshold')
        settings.update(fit_split=split,min_valid_fraction=threshold)
        st.latex(r'\mu_b=N_b^{-1}\sum x_b,\quad \sigma_b=\sqrt{N_b^{-1}\sum(x_b-\mu_b)^2},\quad z_b=(x_b-\mu_b)/s_b')
        st.write('Use only valid TRAIN samples. s=σ unless σ=0, then s=1. [1,3,5] gives μ=3, σ=√(8/3), '
                 'z≈[−1.224745,0,1.224745]. Constant band [2,2,2] maps to zero. The synthetic validation input '
                 'is shifted by +2 but uses the saved training statistics without refitting. A false mask accompanies every '
                 'zero fill; z-scores can exceed [0,1] and change spectral angles, so they are not reflectance or model-ready inputs.')
        st.latex(r'T_{patch}=T_{parent}\,\mathrm{Translation}(c_0,r_0),\quad f=\mathrm{mean}_{r,c}(\bigwedge_b m_b)')
        st.write('A 2×2 patch with three common-valid pixels has f=.75. The canonical acceptance rule is f≥threshold: '
                 'threshold1 rejects it, .75 accepts it while preserving its mask. Full nonoverlapping patches only; incomplete margins are reported. '
                 'At column2,row0, the example patch corner is (120,200). Split scenes before patch extraction.')
        st.graphviz_chart('digraph {rankdir=LR; "Train + mask" -> "Fit mean / population std" -> "Frozen statistics"; "Validation + mask" -> Apply; "Frozen statistics" -> Apply -> "Z-scores + mask" -> "Patch validity / transform";}')
        st.caption('Failure demonstrations: validation fitting is rejected. An all-invalid training band also fails. Split labels need truthful provenance, not just a guard.')
        compute = lambda:normalization(split,threshold)
    elif ident == 'MP34':
        scale = st.selectbox('Synthetic downsampling factor', [2,4],key='MP34_scale')
        sigma = st.selectbox('Gaussian sigma in HR pixels', [0.,.8],key='MP34_sigma')
        invalid = st.checkbox('Mark one HR pixel invalid (must reject)',key='MP34_invalid')
        settings.update(scale=scale,sigma=sigma,noise=0.,seed=34,invalid=invalid)
        st.latex(r'y=D_s(k*x)+n,\quad (H_L,W_L)=(H/s,W/s),\quad T_L=T_H\,\mathrm{Scale}(s,s)')
        st.write('x is the 4×4 single-band ramp0…15; s is 2 or4. With σ=0, blur is disabled: the top-left2×2 '
                 'area mean is (0+1+4+5)/4=2.5. σ=.8 uses a normalized Gaussian with radius ceil(3σ), replicate boundaries, '
                 'then nonoverlapping area averaging. Noise is fixed at0, with seed34 recorded. No clipping, cropping or silent resizing. '
                 'At s=2, the first HR/LR centers are (105,195)/(110,190); exterior corners agree. This is synthetic degradation, not a measured sensor PSF.')
        st.graphviz_chart('digraph {rankdir=LR; "Fully valid HR + transform" -> Gaussian -> "Area mean" -> "LR + scaled transform"; "HR exterior corners" -> "Footprint equality"; "LR + scaled transform" -> "Footprint equality";}')
        st.caption('Invalid HR is rejected before blur so nodata cannot contaminate neighbors. Nondivisible shapes fail. Equal footprint does not prove cross-sensor registration or spectral compatibility.')
        compute = lambda:pairing(scale,sigma,invalid)
    elif ident == 'MP35':
        buffer = st.selectbox('Angular teaching buffer (degrees)', [.02,1.],key='MP35_buffer')
        repeated = st.checkbox('Reuse region a across splits (must reject)',key='MP35_region')
        settings.update(buffer_degrees=buffer,repeated_region=repeated)
        st.latex(r'g=\max(B_w-A_e,A_w-B_e,B_s-A_n,A_s-B_n),\quad g>b')
        st.write('A=[0,0,1,1] is synthetic train region a; B=[2,0,3,1] is synthetic test region b. '
                 'w/e/s/n are west/east/south/north box edges in WGS84 degrees; b is the angular buffer. '
                 'Their maximum axis clearance g=1°. This passes b=.02 but fails b=1 because equality is rejected. '
                 'Cross-split region sets must also be disjoint, even for distant boxes. No frozen test metadata is opened.')
        st.warning('This canonical MP35 axis-envelope teaching audit uses DEGREES, not metres or geodesic distance. '
                   'It is not protocol-v2’s declared 5 km footprint buffer (metric projection/geodesic method) and does not certify that protocol. '
                   'Longitude degrees have latitude-dependent physical size. Antimeridian-crossing boxes are unsupported.')
        st.graphviz_chart('digraph {rankdir=LR; "Assign independent regions" -> "Audit IDs and bounds" -> "Cross-split angular clearance" -> "Only then tile";}')
        st.caption('Patches from one scene remain correlated, not independent geographic replicates. Audit failures keep the split unchanged; this tool never searches for favorable test assignments.')
        compute = lambda:geography(buffer,repeated)
    else:
        offset = st.selectbox('Hypothetical displacement (metres)', [5.,2.5,0.],key='MP36_offset')
        gain = st.selectbox('Toy calibration gain a', [1.1,1.],key='MP36_gain')
        settings.update(offset_metres=offset,gain=gain,bias=.01)
        st.latex(r's=d_L/d_H,\quad \delta_{pixels}=\delta_{metres}/d_H,\quad y=ax+b,\quad e=(a-1)x+b')
        st.latex(r'MSE_{toy}=N^{-1}\sum e_i^2,\quad B=\frac{\sum_i r_iw_i}{\sum_iw_i}')
        st.write('dL,dH are ground sampling distances in metres/pixel. Ratios10/5=2,20/5=4 and10/1.5≈6.667 '
                 'test geometry only. A hypothetical5 m displacement on a5 m grid is1 HR pixel;2.5 m is0.5 pixel. '
                 'For x=[.2,.4], a=1.1,b=.01, residuals=[.03,.05] and toy MSE=.0017. This is not an SR reference score. '
                 'For toy equal-width bins r=[.2,.8], weights [1,0],[0,1],[1,1] give .2,.8,.5. '
                 'Real sensor responses need measured wavelength-aware integration; these weights are only examples.')
        st.graphviz_chart('digraph {rankdir=LR; "Version / license" -> "Bands + time + masks" -> "Registration + calibration" -> "Eligibility manifest" -> "Authorized paired evaluation";}')
        st.warning('Original SEN2VENµS method: 15 m is the SIFT match-rejection threshold, NOT a measured residual alignment error or guarantee. '
                   'The published correction changed the deposit URL, not the scientific conclusions. This review is already complete; it is not repeated here.')
        table('Version-specific applicability and remaining gates',[
            {'record':'Original construction / paper','status':'Method reviewed','limit':'Blue-channel translation, valid SIFT matches, bicubic resampling and pairwise radiometric adjustment; no new residual measurement'},
            {'record':'SEN2VENµS v2.0.0 record14603764','status':'Metadata reviewed; archive uninspected','limit':'Original-method lineage must be checked against versioned files, timestamps, masks and calibration'},
            {'record':'v2 SWIR B11/B12','status':'No VENµS HR reference','limit':'Do not compute or fabricate VENµS reference scores for these bands'},
            {'record':'Real paired acceptance','status':'Gated','limit':'Versioned file inspection, eligibility manifest, registration/calibration checks and resource authorization'}])
        st.caption('Provider harmonization is not untouched sensor truth. One inspected site would not establish geographic generalization. No download is requested or performed by Run.')
        compute = lambda:feasibility(offset,gain)
    st.markdown('References: [Rasterio georeferencing](https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html), '
                '[Raster masks](https://rasterio.readthedocs.io/en/stable/topics/masks.html), '
                '[SEN2VENµS v2.0.0](https://zenodo.org/records/14603764), '
                '[Original paper](https://doi.org/10.3390/data7070096), '
                '[Published URL correction](https://doi.org/10.3390/data8030051). '
                'Version/method statements follow the previously reviewed feasibility record; no new provider investigation is claimed.')
    if not st.button('Run '+ident,key=ident+'_run'):
        return
    started = time.perf_counter()
    try:
        result = compute()
    except (ValueError,OSError,KeyError) as exc:
        st.error('Teaching request rejected: '+str(exc))
        st.caption('Correct the input setting and Run again. No previous numerical result is retained.')
        return
    if ident == 'MP32':
        bands('Input',result['image']); bands('Validity (true = observed)',result['mask'])
        table('Per-band valid statistics (empty = unavailable)',result['statistics'])
        st.write('Coordinate example:',result['center'],'center;',result['corner'],'corner;',result['coordinate_example'])
        st.write('CRS and affine:',result['crs'],result['transform'])
        csv = result['statistics']
    elif ident == 'MP33':
        bands('Training input',result['input']); bands('Mask',result['mask'])
        table('Saved training statistics',result['training_statistics']['bands'])
        bands('Training z-score',result['normalized']); bands('Validation input (+2)',result['validation_input'])
        bands('Validation using TRAIN statistics',result['validation_output'])
        st.write('Accepted patch records:',result['accepted_patches'],'; margins:',result['margins'])
        st.write('Separate translated patch corner:',result['translated_patch_corner'])
        st.caption('A constant training band has scale1; shifted validation values stay shifted. Recentring validation would hide that change.')
        csv = result['training_statistics']['bands']
    elif ident == 'MP34':
        bands('HR',result['hr']); bands('LR',result['lr']); bands('HR validity',result['valid'])
        st.write('Declared operator, shapes, transforms and corners:',result['metadata'])
        st.write('HR/LR first centers:',result['hr_center'],result['lr_center'])
        st.caption('Area averaging reduces sample count. Matching exterior corners checks this synthetic footprint, not physical sensor agreement.')
        csv = result['lr'][0]
    elif ident == 'MP35':
        table('Synthetic scene declarations',result['scenes'])
        table('Cross-split angular clearance',result['audit']['cross_split_pairs'])
        st.write(result['audit']['rule']); st.write('Counts:',result['audit']['counts'])
        csv = result['audit']['cross_split_pairs']
    else:
        table('Geometric ratios',result['ratios'])
        table('Toy radiometric calculation',{'x':result['reference'],'y':result['prediction'],'residual':result['residual']})
        table('Toy spectral integration',{'weights':result['weights'],'response':result['toy_responses']})
        st.write('Hypothetical offset in HR pixels:',result['offset_hr_pixels'],'; toy calibration MSE:',result['toy_mse'])
        st.caption('These calculations identify potential confounders; they neither measure registration residuals nor establish eligibility for real paired reference metrics.')
        csv = result['ratios']
    sources = list((root/'stage_08_remote_sensing').glob('MP*/mp*_tools.py'))
    sources += [root/'app/remote_sensing_labs.py',root/'app/remote_sensing_native_ui.py',
                root/'src/rssr/core.py',root/'src/rssr/data.py',root/'reports/mp36-feasibility.md']
    settings.update(seconds=time.perf_counter()-started,numpy=np.__version__,torch=torch.__version__,
                    source_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    st.download_button('Download result and provenance',json.dumps({'settings':settings,'result':result},indent=2,allow_nan=False),
                       ident+'-lesson.json','application/json',key=ident+'_json')
    st.download_button('Download numerical table',pd.DataFrame(csv).to_csv(index=False),ident+'-table.csv','text/csv',key=ident+'_csv')
    st.success(f'{ident} synthetic computation completed in {settings["seconds"]:.4f} s. Scientific acceptance remains separate.')
