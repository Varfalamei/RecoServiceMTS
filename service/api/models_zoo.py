from abc import ABC
from typing import List


class BaseModel(ABC):
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
        pass


class DumpModel(BaseModel):
    def __init__(self):
        super().__init__()
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
        reco = list(range(k_recs))
        return reco
