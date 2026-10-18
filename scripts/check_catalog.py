import json
from pathlib import Path
from gradio_client import Client
result=Client('http://127.0.0.1:7860',verbose=False).predict(api_name='/catalog')
assert result['mode'].startswith('cached metadata')
assert result['last_complete_refresh'] is not None or result['seconds_since_complete_refresh'] is None
assert len(result['observations'])<=result['observation_limit']
root=Path(__file__).resolve().parents[1]
(root/'reports/catalog-http.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Read-only catalog HTTP response validated')
