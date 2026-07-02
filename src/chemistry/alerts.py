"""Heuristic SMARTS alerts for MVP saliency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlertPattern:
    """A SMARTS alert and medicinal chemistry rationale."""

    name: str
    smarts: str
    reason: str
    toxic_alert: bool = False


ALERT_PATTERNS: tuple[AlertPattern, ...] = (
    AlertPattern("halogen", "[F,Cl,Br,I]", "할로젠은 lipophilicity와 hERG liability를 높일 수 있습니다."),
    AlertPattern("aromatic_ring", "a1aaaaa1", "방향족 고리는 lipophilicity와 대사 안정성에 영향을 줄 수 있습니다."),
    AlertPattern("long_alkyl_chain", "[CX4][CX4][CX4][CX4]", "긴 alkyl chain은 용해도를 낮출 수 있습니다."),
    AlertPattern("basic_amine", "[NX3;H2,H1,H0;!$(NC=O)]", "염기성 amine은 BBB 투과와 hERG risk에 영향을 줄 수 있습니다."),
    AlertPattern("nitro_group", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]", "nitro group은 대표적인 structural toxicity alert입니다.", True),
    AlertPattern("aniline_like", "a[NH2,NH1,NH0]", "aniline-like motif는 AMES risk와 연관될 수 있습니다.", True),
)

