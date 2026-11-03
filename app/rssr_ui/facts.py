"""Read-only adapters: facts retain their scope and named source."""
from dataclasses import asdict, dataclass
from rssr.scaling import SCALES
from rssr.model_availability import availability


@dataclass(frozen=True)
class SiteFact:
    id: str
    value: int
    unit: str
    status: str
    source: str
    limitations: str
    commit: str | None = None
    generated_at: str | None = None


def site_facts(root, index):
    projects = index['projects']
    return [
        SiteFact('families', len(availability(root)), 'configured model families', 'VERIFIED', 'rssr.model_availability.availability', 'Registry membership is not matched scientific validation.'),
        SiteFact('scales', len(SCALES), 'configured scale choices', 'VERIFIED', 'rssr.scaling.SCALES', 'Host preflight and checkpoint compatibility still apply; native ×2 cascades are not detail recovery.'),
        SiteFact('lessons', len(projects), 'registered lessons', 'VERIFIED', 'rssr.evidence.build_index/projects', 'Native interaction coverage is separate from every teaching criterion and learner mastery.'),
        SiteFact('stages', len({p['stage'] for p in projects}), 'registered stages', 'VERIFIED', 'rssr.evidence.build_index/projects', 'No learner completion inferred.'),
    ]


def fact_records(root, index):
    return [asdict(fact) for fact in site_facts(root, index)]
