from abc import ABC
from typing import List, Set

import pandas as pd


class BaseModelZoo(ABC):
    def __init__(self):
        pass

    @staticmethod
    def unique(items: List[int]) -> List[int]:
        seen: Set[int] = set()
        seen_add = seen.add
        return [item for item in items if not (item in seen or seen_add(item))]

    def reco_predict(
        self,
        user_id: int,
        k_recs: int
    ) -> List[int]:
        """
        Main function for recommendation items to users
        :param user_id: user identification
        :param k_recs: how many recs do you need
        :return: list of recommendation ids
        """


class DumpModel(BaseModelZoo):
    def reco_predict(
        self,
        user_id: int,
        k_recs: int
    ) -> List[int]:
        """
        Main function for recommendation items to users
        :param user_id: user identification
        :param k_recs: how many recs do you need
        :return: list of recommendation ids
        """
        reco = list(range(k_recs))
        return reco


class TopPopularAllCovered(BaseModelZoo):
    def reco_predict(
        self,
        user_id: int,
        k_recs: int
    ) -> List[int]:
        """
        Main function for recommendation items to users
        :param user_id: user identification
        :param k_recs: how many recs do you need
        :return: list of recommendation ids
        """
        reco = [
            10440, 15297, 9728, 13865, 2657,
            4151, 3734, 6809, 4740, 4880, 7571,
            11237, 8636, 14741
        ][:k_recs]
        return reco


class Popular(BaseModelZoo):
    def reco_predict(
        self,
        user_id: int,
        k_recs: int
    ) -> List[int]:
        """
        Main function for recommendation items to users
        :param user_id: user identification
        :param k_recs: how many recs do you need
        :return: list of recommendation ids
        """
        reco = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]
        return reco


class KNNModelWithTop(BaseModelZoo):
    def __init__(
        self,
        path_to_reco: str = "data/BlendingKNNWithAddFeatures.csv.gz"
    ):
        super().__init__()
        self.path_to_reco = path_to_reco
        if self.path_to_reco.endswith('csv.gz'):
            self.data = pd.read_csv(path_to_reco, compression='gzip')
        elif self.path_to_reco.endswith('.csv'):
            self.data = pd.read_csv(path_to_reco)
        self.top_reco = [
            10440, 15297, 9728, 13865, 3734, 12192, 4151, 11863, 7793, 7829
        ]

    def reco_predict(
        self,
        user_id: int,
        k_recs: int
    ) -> List[int]:
        """
        Main function for recommendation items to users
        :param user_id: user identification
        :param k_recs: how many recs do you need
        :return: list of recommendation ids
        """
        reco = (
            self.data[self.data.user_id == user_id]
            .item_id
            .tolist()
            [:k_recs]
        )

        if len(reco) < k_recs:
            reco.extend(self.top_reco)
            reco = self.unique(reco)[:k_recs]  # Удаляем дубли

        return reco
