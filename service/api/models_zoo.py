from abc import ABC
from typing import List

import pandas as pd


class BaseModelZoo(ABC):
    def __init__(self):
        pass

    @staticmethod
    def unique(items: List[int]) -> List[int]:
        seen = set()
        return [item for item in items if not (item in seen or seen.add(item))]

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


class UserKnnTfIdfTop(BaseModelZoo):
    def __init__(self):
        super(UserKnnTfIdfTop).__init__()
        self.data = pd.read_csv('data/UserKnnTfIdf.csv')
        self.top_reco = [
            10440, 15297, 9728, 13865, 4151,
            3734, 2657, 4880, 142, 6809
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


class ItemKNN(BaseModelZoo):
    def __init__(self):
        super(ItemKNN).__init__()
        self.data = pd.read_csv('data/ItemKNN.csv')
        self.top_reco = [
            10440, 15297, 9728, 13865, 4151,
            3734, 2657, 4880, 142, 6809
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
