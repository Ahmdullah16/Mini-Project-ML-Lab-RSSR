from contextlib import contextmanager
import pytest
from rssr.external_storage import ExternalStore,from_environment


class Objects:
    def put_object(self,**kwargs): self.value=kwargs
    def head_object(self,**kwargs):
        return {'ContentLength':len(self.value['Body']),'Metadata':self.value['Metadata']}


class Database:
    def __init__(self): self.rows={}; self.fail=False
    @contextmanager
    def transaction(self):
        old=self.rows.copy()
        try: yield
        except Exception: self.rows=old; raise
    def execute(self,sql,args=None):
        if self.fail: raise RuntimeError('fixture database unavailable')
        if args: self.rows.setdefault(args[0],args)


def test_idempotency_and_failed_database_not_success():
    objects=Objects(); db=Database(); store=ExternalStore(objects,db,'fixture-bucket')
    a=store.persist(b'example',{'item':'fixture'})
    assert a==store.persist(b'example',{'item':'fixture'}) and len(db.rows)==1
    db.fail=True
    with pytest.raises(RuntimeError): store.persist(b'new',{'item':'other'})
    assert len(db.rows)==1


def test_missing_configuration_does_not_connect(monkeypatch):
    for key in ['RSSR_S3_ENDPOINT','RSSR_S3_BUCKET','RSSR_S3_ACCESS_KEY_ID','RSSR_S3_SECRET_ACCESS_KEY','RSSR_DATABASE_URL']:
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(ValueError,match='Missing external storage settings'): from_environment()
