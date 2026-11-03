"""Single source for scoped HTML surfaces and chart colours."""
PALETTES = {
    'Light': dict(bg='#F5F8FC', card='#FFFFFF', text='#14253D', muted='#4C617A', line='#66758A', accent='#086B83', soft='#E5F3F7'),
    'Dark': dict(bg='#0B1220', card='#172339', text='#F8FAFC', muted='#B8C6DB', line='#8495AE', accent='#67E8F9', soft='#20384B'),
}


def contrast(a, b):
    def luminance(color):
        values = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
        linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in values]
        return sum(v*w for v, w in zip(linear, (.2126, .7152, .0722)))
    x, y = sorted((luminance(a), luminance(b)))
    return (y+.05)/(x+.05)


def css(mode, reduced=True):
    p = PALETTES[mode]
    variables = ';'.join(f'--lab-{key}:{value}' for key, value in p.items())
    return '''<style>
:root{VARIABLES;}
.lab-hero,.lab-grid,.lab-profile,.lab-footer,.aurora-banner{color:var(--lab-text);font-family:system-ui,sans-serif;}
.lab-hero{background:var(--lab-bg);padding:clamp(24px,5vw,56px);border:1px solid var(--lab-line);border-radius:20px;margin:16px 0 24px;position:relative;overflow:hidden;}
.lab-hero::before{content:"";display:block;height:3px;background:linear-gradient(90deg,#22D3EE,#6366F1,#E879F9);margin-bottom:24px;}
.lab-hero h1{color:var(--lab-text);font-size:clamp(2rem,4vw,3.6rem);line-height:1.1;letter-spacing:-.04em;max-width:900px;}
.lab-hero p{color:var(--lab-muted);line-height:1.7;max-width:740px;}
.lab-kicker,.lab-number{color:var(--lab-accent);font-size:.8rem;font-weight:700;letter-spacing:.1em;}
.lab-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:16px;margin:16px 0 24px;}
.lab-card,.lab-profile{background:var(--lab-card);border:1px solid var(--lab-line);border-radius:16px;padding:24px;min-width:0;overflow-wrap:anywhere;}
.lab-card h3{color:var(--lab-text);font-size:1.15rem;}
.lab-card p,.lab-profile p,.lab-footer{color:var(--lab-muted);line-height:1.7;}
.lab-pills{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0;}
.lab-pill{background:var(--lab-card);color:var(--lab-text);border:1px solid var(--lab-line);border-radius:24px;padding:8px 12px;font-size:.8rem;}
.lab-profile a,.lab-footer a,.aurora-skip{color:var(--lab-accent);text-decoration:underline;}
.lab-footer{background:var(--lab-bg);padding:24px;border-top:1px solid var(--lab-line);margin-top:32px;}
.aurora-skip{display:inline-block;background:var(--lab-card);padding:12px;min-height:44px;}
.aurora-skip:focus-visible,.lab-profile a:focus-visible,.lab-footer a:focus-visible{outline:3px solid var(--lab-accent);outline-offset:3px;}
.aurora-banner{background:var(--lab-card);border-left:4px solid var(--lab-accent);padding:16px;margin:16px 0;}
@media(max-width:480px){.lab-card,.lab-profile{padding:16px;}.lab-hero{padding:24px;}}
@media(prefers-reduced-motion:reduce){.lab-hero,.lab-card{animation:none!important;transition:none!important;}}
MOTION
</style>'''.replace('VARIABLES', variables).replace('MOTION', '' if reduced else '.lab-card{transition:transform .2s ease;}@media(hover:hover){.lab-card:hover{transform:translateY(-2px);}}')


def chart_theme(mode):
    p = PALETTES[mode]
    return {'background': p['card'], 'axis': {'labelColor': p['text'], 'titleColor': p['text']},
            'legend': {'labelColor': p['text'], 'titleColor': p['text']}}
