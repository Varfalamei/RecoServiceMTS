from abc import ABC
from typing import List


class BaseModelZoo(ABC):
    def __init__(self):
        pass

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
