# AURORA X foundation

VERIFIED: app/rssr_ui/tokens.py owns surface colours and chart-theme adapter. System fonts; no CDN, parent JavaScript, image enhancement or private Streamlit selectors. Native global theme uses Streamlit Settings; Appearance selects owned panels. Synchronized global theming is NOT VERIFIED.

8px spacing,16–20px cards, one cyan/indigo/fuchsia hero anchor, navy/slate dark panels. Existing native focus/labels remain. Registry card strings are escaped. Responsive grid collapses to one column. Hover motion is opt-in; browser reduced-motion, Reduce motion, or lite=1 disables it. Images receive no decorative filters.

Text/muted/accent on both card/background pairs meet4.5:1 and border pairs meet3:1 in tests. This is not full WCAG2.2AA certification. Full manual keyboard/screen-reader review remains NOT VERIFIED.

Component reuse: Hero/ModelFamilyCard/ResearcherCard/Footer in research_design; shell/EvidenceDrawer in rssr_ui.shell; SiteFact in rssr_ui.facts; ExperimentPassport in rssr_ui.passport; existing comparison and metric modules retained. No fake controls for absent features. Developer gallery includes hero/cards and warning/empty-state fixtures. Complete workflow rail and every state screenshot: NOT VERIFIED.
