from abc import ABC
from abc import abstractmethod

class TransactionCostModel(ABC):
    """
    Abstract transaction cost model.
    """

    @abstractmethod
    def cost(
        self,
        turnover: float,
    ) -> float:
        """
        Compute transaction cost.
        """

class FixedCostModel(
    TransactionCostModel,
):
    """
    Fixed commission model.
    """

    def __init__(
    self,
    *models: TransactionCostModel,
):

        if len(models) == 0:
            raise ValueError(
            "At least one cost model is required."
        )

        for model in models:

            if not isinstance(
            model,
            TransactionCostModel,
        ):
             raise TypeError(
                "All models must inherit "
                "TransactionCostModel."
            )

        self.models = models

    def cost(
        self,
        turnover: float,
    ) -> float:

        return self.commission


class ProportionalCostModel(
    TransactionCostModel,
):
    """
    Percentage transaction cost.
    """

    def __init__(
        self,
        rate: float,
    ):

        if rate < 0:
            raise ValueError(
                "rate must be non-negative."
            )

        self.rate = rate

    def cost(
        self,
        turnover: float,
    ) -> float:

        return turnover * self.rate

class BidAskSpreadModel(
    TransactionCostModel,
):
    """
    Half-spread transaction model.
    """

    def __init__(
        self,
        spread: float,
    ):

        if spread < 0:
            raise ValueError(
                "spread must be non-negative."
            )

        self.spread = spread

    def cost(
        self,
        turnover: float,
    ) -> float:

        return turnover * self.spread

class CompositeCostModel(
    TransactionCostModel,
):
    """
    Combine multiple transaction cost models.
    """

    def __init__(
        self,
        *models,
    ):

        self.models = models

    def cost(
        self,
        turnover: float,
    ) -> float:

        return sum(
            model.cost(turnover)
            for model in self.models
        )

