"""Versioned local SQLite catalog; one writer, explicit foreign keys."""
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_versions(version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT NOT NULL,version TEXT NOT NULL,source_url TEXT NOT NULL,license TEXT NOT NULL,citation TEXT NOT NULL,manifest_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL REFERENCES datasets(id),provider_item_id TEXT NOT NULL,region_id TEXT NOT NULL,sensor TEXT NOT NULL,acquisition_time TEXT,availability_time TEXT,ingestion_time TEXT NOT NULL,crs TEXT,transform TEXT,bounds TEXT,bands TEXT,units TEXT,nodata TEXT,quality_flags TEXT,source_location TEXT NOT NULL,checksum TEXT NOT NULL,UNIQUE(dataset_id,provider_item_id));
CREATE TABLE IF NOT EXISTS pairs_or_tiles(id TEXT PRIMARY KEY,split TEXT NOT NULL CHECK(split IN ('train','validation','test','inference')),lr_path TEXT NOT NULL,hr_path TEXT,scale INTEGER NOT NULL CHECK(scale IN (2,4)),crop_bounds TEXT,alignment_method TEXT,degradation_config TEXT,checksums TEXT);
CREATE TABLE IF NOT EXISTS pair_sources(pair_id TEXT REFERENCES pairs_or_tiles(id),scene_id TEXT REFERENCES scenes(id),PRIMARY KEY(pair_id,scene_id));
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,git_commit TEXT NOT NULL,dataset_manifest_hash TEXT NOT NULL,configuration_hash TEXT NOT NULL,model TEXT NOT NULL,seed INTEGER NOT NULL,environment TEXT NOT NULL,hardware TEXT NOT NULL,start_time TEXT NOT NULL,end_time TEXT,status TEXT NOT NULL CHECK(status IN ('running','completed','failed','interrupted')),checkpoint_path TEXT,checkpoint_hash TEXT);
CREATE TABLE IF NOT EXISTS metrics(run_id TEXT REFERENCES runs(id),scene_or_tile_id TEXT REFERENCES pairs_or_tiles(id),metric_name TEXT NOT NULL,value REAL,units TEXT,protocol_version TEXT NOT NULL,evaluation_split TEXT NOT NULL,PRIMARY KEY(run_id,scene_or_tile_id,metric_name));
CREATE TABLE IF NOT EXISTS artifacts(run_id TEXT REFERENCES runs(id),artifact_type TEXT NOT NULL,path TEXT NOT NULL,checksum TEXT NOT NULL,PRIMARY KEY(run_id,path));
CREATE TABLE IF NOT EXISTS ingestion_jobs(id TEXT PRIMARY KEY,provider TEXT NOT NULL,query_config TEXT NOT NULL,start_time TEXT NOT NULL,end_time TEXT,status TEXT NOT NULL,items_found INTEGER NOT NULL DEFAULT 0,items_added INTEGER NOT NULL DEFAULT 0,failures TEXT,cursor_or_watermark TEXT);
"""


def connect(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute('CREATE TABLE IF NOT EXISTS schema_versions(version INTEGER PRIMARY KEY)')
    versions = [r[0] for r in db.execute("SELECT version FROM schema_versions")]
    if versions not in ([],[1],[2]):
        db.close(); raise ValueError("Unsupported schema migration version")
    db.executescript(SCHEMA)
    # Additive v1 -> v2 migration; existing experiment rows are untouched.
    with db:
        db.execute('''CREATE TABLE IF NOT EXISTS inference_jobs(
            id TEXT PRIMARY KEY, provider_item_id TEXT NOT NULL, acquisition_time TEXT,
            ingestion_time TEXT NOT NULL, inference_time TEXT, status TEXT NOT NULL,
            source_path TEXT, source_sha256 TEXT, checkpoint_sha256 TEXT NOT NULL,
            output_path TEXT, output_sha256 TEXT, provenance TEXT NOT NULL, failure TEXT)''')
        db.execute('DELETE FROM schema_versions')
        db.execute('INSERT INTO schema_versions VALUES(2)')
    return db


def export(db, directory):
    import csv
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for table in ("datasets","scenes","pairs_or_tiles","pair_sources","runs","metrics","artifacts","ingestion_jobs","inference_jobs"):
        cur = db.execute(f"SELECT * FROM {table}")
        names = [d[0] for d in cur.description]; rows = cur.fetchall()
        with (directory/f"{table}.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(names); writer.writerows(rows)
        (directory/f"{table}.json").write_text(json.dumps([dict(zip(names,row)) for row in rows],indent=2),encoding="utf-8")
