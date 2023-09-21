"""

Predict formula blocks in a page.

@kylel

"""

from typing import Dict, Optional

from papermage.predictors.base_predictors.lp_predictors import LPPredictor


class LPEffDetFormulaPredictor(LPPredictor):
    @classmethod
    def from_pretrained(
        cls,
        device: Optional[str] = None,
    ):
        return super().from_pretrained(
            config_path="lp://efficientdet/MFD",
            device=device,
        )
