"""Optional operator-managed S3/PostgreSQL persistence, disabled in the demo.

This adapter stores content-addressed artifacts and provenance. It does not claim
to migrate the complete normalized SQLite research schema to PostgreSQL.
"""
import hashlib
import json
import os


class ExternalStore:
    def __init__(self,objects,database,bucket,prefix='rssr'):
        if not bucket or not prefix or any(v in prefix for v in ['..','\\']):
            raise ValueError('Explicit bucket and safe object prefix required')
        self.objects,self.database,self.bucket,self.prefix=objects,database,bucket,prefix.strip('/')

    def persist(self,content,provenance,content_type='image/tiff'):
        """Upload first; commit metadata second. Retry uses the same content key.

        A database failure can leave an unreferenced object; it must not produce a
        successful record. Never log connection strings or credentials.
        """
        digest=hashlib.sha256(content).hexdigest()
        key=f'{self.prefix}/sha256/{digest}'
        encoded=json.dumps(provenance,sort_keys=True,allow_nan=False)
        event=hashlib.sha256((digest+encoded).encode()).hexdigest()
        self.objects.put_object(Bucket=self.bucket,Key=key,Body=content,
                                ContentType=content_type,Metadata={'sha256':digest})
        verified=self.objects.head_object(Bucket=self.bucket,Key=key)
        if verified['ContentLength']!=len(content) or verified.get('Metadata',{}).get('sha256')!=digest:
            raise ValueError('External object acknowledgement mismatch')
        with self.database.transaction():
            self.database.execute('''CREATE TABLE IF NOT EXISTS rssr_external_artifacts (
                event_id TEXT PRIMARY KEY, object_key TEXT NOT NULL, sha256 TEXT NOT NULL,
                provenance JSONB NOT NULL, stored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
            self.database.execute('''INSERT INTO rssr_external_artifacts(event_id,object_key,sha256,provenance)
                VALUES (%s,%s,%s,%s::jsonb) ON CONFLICT (event_id) DO NOTHING''',
                (event,key,digest,encoded))
        return {'event_id':event,'object_key':key,'sha256':digest,'bucket':self.bucket}


def from_environment():
    """Explicit opt-in; SDK imports and connections happen only when requested."""
    required=['RSSR_S3_ENDPOINT','RSSR_S3_BUCKET','RSSR_S3_ACCESS_KEY_ID',
              'RSSR_S3_SECRET_ACCESS_KEY','RSSR_DATABASE_URL']
    missing=[key for key in required if not os.environ.get(key)]
    if missing: raise ValueError('Missing external storage settings: '+', '.join(missing))
    if not os.environ['RSSR_S3_ENDPOINT'].startswith('https://'):
        raise ValueError('External object storage requires HTTPS')
    import boto3
    import psycopg
    objects=boto3.client('s3',endpoint_url=os.environ['RSSR_S3_ENDPOINT'],
                         region_name=os.environ.get('RSSR_S3_REGION','us-east-1'),
                         aws_access_key_id=os.environ['RSSR_S3_ACCESS_KEY_ID'],
                         aws_secret_access_key=os.environ['RSSR_S3_SECRET_ACCESS_KEY'])
    database=psycopg.connect(os.environ['RSSR_DATABASE_URL'],connect_timeout=10,sslmode='require')
    return ExternalStore(objects,database,os.environ['RSSR_S3_BUCKET'])
