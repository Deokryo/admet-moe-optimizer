"""Abnormal endpoint gate."""

from __future__ import annotations

from dataclasses import dataclass

from src.predictors.base import Prediction


@dataclass(frozen=True)
class AbnormalityConfig:
    """Thresholds and target context for endpoint gating."""

    is_cns_target: bool
    herg_threshold: float = 0.55
    ames_threshold: float = 0.50
    min_solubility: float = -3.5
    min_logp: float = 1.0
    max_logp: float = 3.0


@dataclass(frozen=True)
class Abnormality:
    """Flagged endpoint with reason."""

    endpoint: str
    value: float
    severity: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize for display."""
        return {
            "Endpoint": self.endpoint,
            "예측값": round(self.value, 4),
            "심각도": self.severity,
            "사유": self.reason,
        }


class AbnormalityGate:
    """Apply rule-based abnormality checks to descriptor and prediction outputs."""

    def __init__(self, config: AbnormalityConfig) -> None:
        """Initialize with thresholds."""
        self.config = config

    def evaluate(self, descriptors: dict[str, float], predictions: dict[str, Prediction]) -> list[Abnormality]:
        """Return endpoint abnormalities."""
        flags: list[Abnormality] = []
        sol = float(predictions["Solubility Expert"].value)
        logp = float(predictions["Lipophilicity Expert"].value)
        bbb = float(predictions["BBB Expert"].value)
        herg = float(predictions["hERG Expert"].value)
        ames = float(predictions["AMES Expert"].value)

        if sol < self.config.min_solubility:
            gap = self.config.min_solubility - sol
            flags.append(
                Abnormality(
                    "Solubility",
                    sol,
                    "높음",
                    f"예측 용해도가 목표 minimum {self.config.min_solubility:.2f}보다 {gap:.2f} 낮습니다.",
                )
            )
        if logp < self.config.min_logp:
            gap = self.config.min_logp - logp
            flags.append(
                Abnormality(
                    "Lipophilicity",
                    logp,
                    "중간",
                    f"LogP가 목표 범위 {self.config.min_logp:.2f}-{self.config.max_logp:.2f}보다 {gap:.2f} 낮습니다.",
                )
            )
        elif logp > self.config.max_logp:
            gap = logp - self.config.max_logp
            flags.append(
                Abnormality(
                    "Lipophilicity",
                    logp,
                    "중간",
                    f"LogP가 목표 범위 {self.config.min_logp:.2f}-{self.config.max_logp:.2f}보다 {gap:.2f} 높습니다.",
                )
            )
        if self.config.is_cns_target and bbb < 0.45:
            flags.append(Abnormality("BBB", bbb, "중간", "CNS 타깃에서는 충분한 BBB 통과 가능성이 필요합니다."))
        if not self.config.is_cns_target and bbb > 0.65:
            flags.append(Abnormality("BBB", bbb, "중간", "비-CNS 타깃에서는 높은 BBB 통과 가능성이 선호되지 않을 수 있습니다."))
        if herg >= self.config.herg_threshold:
            flags.append(Abnormality("hERG", herg, "높음", f"hERG risk 예측값이 임계값 {self.config.herg_threshold:.2f} 이상입니다."))
        if ames >= self.config.ames_threshold:
            flags.append(Abnormality("AMES", ames, "높음", f"AMES risk 예측값이 임계값 {self.config.ames_threshold:.2f} 이상입니다."))

        if descriptors["qed"] < 0.25:
            flags.append(Abnormality("QED", descriptors["qed"], "낮음", "현재 profile에서 QED가 낮습니다."))
        return flags
