from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_course_identity_is_consistent_in_submission_and_site():
    targets = [
        ROOT / "README.md",
        ROOT / "paper/main.tex",
        ROOT / "app/showcase.py",
        ROOT / "docs/student-project-guide.md",
    ]
    for path in targets:
        content = path.read_text(encoding="utf-8")
        assert "264102307" in content
        assert "EE5302L" in content
        assert "Introduction to Machine Learning Lab" in content


def test_beginner_guide_explains_real_forward_flow_and_boundaries():
    guide = (ROOT / "docs/student-project-guide.md").read_text(encoding="utf-8")
    assert "def forward(image):" in guide
    assert "bicubic_resize" in guide
    assert "selected_model_blocks" in guide
    assert "PixelShuffle" in guide
    assert "pseudocode" in guide
    assert "without proving that real ground detail was recovered" in guide
    model_source = (ROOT / "src/rssr/models.py").read_text(encoding="utf-8")
    assert "def forward(self, x):" in model_source
    assert "return base + self.constant_gate*residual" in model_source


def test_overview_loads_the_beginner_guide():
    showcase = (ROOT / "app/showcase.py").read_text(encoding="utf-8")
    assert "docs/student-project-guide.md" in showcase
    assert "Project basics · a simple explanation you can present" in showcase
