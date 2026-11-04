"""Read-only public metadata and explicitly prepared licensed training sample."""
import hashlib
import json
from pathlib import Path


def records(root):
    manifest = json.loads((Path(root)/'configs/naip-development-manifest.json').read_text(encoding='utf-8'))
    rows = [dict(sample_id=s['id'], split=s['split'], acquisition=s['acquisition_time'],
                 region=s['region_id'], sha256=s['sha256'], source_url=s['url'],
                 sensor=manifest['sensor'], bands=manifest['bands'], license=manifest['license'],
                 units=manifest['units'], bbox=None, crs=None, pixel_spacing=None,
                 geometry_status='Not inspected in this metadata catalogue; no place name inferred',
                 provenance='configs/naip-development-manifest.json') for s in manifest['scenes']]
    geometry_path=Path(root)/'configs/public-sample-geometry.json'
    geometry=json.loads(geometry_path.read_text(encoding='utf-8'))['samples'] if geometry_path.exists() else []
    for row in rows:
        for item in geometry:
            if item['sample_id']==row['sample_id'] and item['sha256']==row['sha256']:
                row.update(bbox=item['bbox_wgs84'],crs=item['crs'],pixel_spacing=item['pixel_spacing'],
                           geometry_status=item['geometry_scope'],pixel_spacing_units=item['pixel_spacing_units'])
    return rows


def select(rows, query='', split='All'):
    return [r for r in rows if (split=='All' or r['split']==split) and
            query.casefold() in json.dumps(r,ensure_ascii=False).casefold()]


def render(root, key):
    import streamlit as st
    with st.expander('Explore real data · public sample catalogue'):
        st.caption('Read-only public provenance. NAIP is aerial imagery. Training illustrations are not independent accuracy evidence.')
        st.info('Taj Mahal/Agra and IIT Guwahati/Brahmaputra: imagery and location metadata not yet verified. No substitute is labeled as either place.')
        query=st.text_input('Search public sample metadata',key=key+'query')
        split=st.selectbox('Dataset split',['All','train','validation','test'],key=key+'split')
        rows=select(records(root),query,split)
        if not rows: st.info('No public samples match these filters.')
        else: st.dataframe([{k:r[k] for k in ('sample_id','split','acquisition','region','sensor')} for r in rows],hide_index=True)
        st.download_button('Download filtered public manifest',json.dumps(rows,indent=2),'public-samples.json','application/json',key=key+'manifest')
        st.caption('Original source checksums and licenses are included. No held-out pixels are loaded or evaluated by browsing. Inspected geometry describes the full source HR raster, not the LR preview footprint; uninspected fields remain unknown.')
        if st.button('Load licensed training preview',key=key+'preview'):
            try:
                from inference_ui import BASE_CONFIG,CACHE,fetch_asset
                data=fetch_asset(BASE_CONFIG['sample'],CACHE).read_bytes()
                if hashlib.sha256(data).hexdigest()!=BASE_CONFIG['sample']['sha256']:
                    raise ValueError('Training preview checksum mismatch')
                st.session_state[key+'bytes']=data
            except (OSError,ValueError) :
                st.error('Verified training preview unavailable. Retry when the public asset service is accessible.')
        data=st.session_state.get(key+'bytes')
        if data:
            st.image(data,caption='OpenSR ROI_00006 · training illustration · synthetic ×2 LR · NAIP aerial RGB; USDA/OpenSR attribution. Not an HR reference.')
            st.download_button('Download licensed training LR',data,'naip-training-lr.png','image/png',key=key+'image')
