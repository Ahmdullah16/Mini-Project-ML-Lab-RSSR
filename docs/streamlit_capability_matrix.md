# Streamlit capability matrix — 2026-11-03

Pinned/local Python3.12.14 and Streamlit1.66.0. API presence is not browser acceptance. Set RSSR_DEV_CAPABILITIES=1 and run `streamlit run app/aurora_capability_lab.py` for the separate developer page; disabled by default and never routed publicly.

| Capability | Local evidence | Acceptance |
|---|---|---|
| navigation/Page/page_link and top/grouped signature | Installed signatures inspected | VERIFIED presence; NOT VERIFIED migrated routes/deep links |
| keyed container/html/static SVG | Capability AppTest | VERIFIED local rendering |
| popover/pills/status | Capability AppTest | VERIFIED render; NOT VERIFIED full assistive-tech operation |
| fragment/dialog/toast/context | API presence inspected | NOT VERIFIED behavior; no product reliance |
| query params | lite=1 disables optional motion in AppTest | VERIFIED |
| upload/download | Existing bounded paths and passport guards | VERIFIED affected tests; NOT VERIFIED all hostile-file cases anew |
| Custom JS / parent control | Not enabled | NOT VERIFIED; native fallback retained |
| CSS | Owned classes only; no private selectors | VERIFIED source checks; native controls keep native theme |
| Responsive widths | Overview document width equals1920/1440/1024/768/390 in two panel modes | VERIFIED measurement; NOT VERIFIED full visual matrix |
| Keyboard/screen reader/Cloud cold start | Separate acceptance required | NOT VERIFIED |

Official references checked: [navigation](https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation), [layouts](https://docs.streamlit.io/develop/concepts/design/layouts-and-containers), [authentication](https://docs.streamlit.io/develop/concepts/connections/authentication). Installed probes, not documentation alone, establish local API availability.
