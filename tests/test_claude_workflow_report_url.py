from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claude.yml"
)


def test_claude_review_report_url_uses_current_repository():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "https://${{ github.repository_owner }}.github.io/${{ github.event.repository.name }}/pr/" in workflow
    assert "https://skyaho.github.io/Autoresearch/pr/" not in workflow
