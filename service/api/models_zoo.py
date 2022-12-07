from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import dill
import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender

data_path = Path(__file__).parent.parent.parent/"data"


class BaseModelZoo(ABC):
    def __init__(self):
        pass

    @staticmethod
    def unique(items: List[int]) -> List[int]:
        seen: Set[int] = set()
        seen_add = seen.add
        return [item for item in items if not (item in seen or seen_add(item))]

    @abstractmethod
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


class KNNModelBM25(BaseModelZoo):
    def __init__(
        self,
        path_to_model: str = "data/knn_bm25.pickle"
    ):
        super().__init__()

        with open(path_to_model, 'rb') as f:
            self.model = dill.load(f)

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
        try:
            reco = self.model.predict(user_id=user_id)
        except KeyError:
            reco = []

        if len(reco) < k_recs:
            reco.extend(self.top_reco)
            reco = self.unique(reco)[:k_recs]  # Удаляем дубли

        return reco


class UserKNN:
    """
    Class for fit-perdict UserKNN model and BM25
    based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(
        self,
        dist_model: ItemItemRecommender,
        n_neighbors: int = 50,
        verbose: int = 1,
    ):
        self.n_neighbors = n_neighbors
        self.dist_model = dist_model
        self.verbose = verbose
        self.is_fitted = False

        self.mapping: Dict[str, Dict[int, int]] = defaultdict(dict)

        self.weights_matrix = None
        self.users_watched = None

    def get_mappings(self, train):
        self.mapping['users_inv_mapping'] = dict(
            enumerate(train['user_id'].unique())
        )
        self.mapping['users_mapping'] = {
            v: k for k, v in self.mapping['users_inv_mapping'].items()
        }

        self.mapping['items_inv_mapping'] = dict(
            enumerate(train['item_id'].unique())
        )
        self.mapping['items_mapping'] = {
            v: k for k, v in self.mapping['items_inv_mapping'].items()
        }

    def get_matrix(
        self, df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        weight_col: str = None,
    ):
        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        if hasattr(self.mapping['users_mapping'], 'get') and \
                hasattr(self.mapping['items_mapping'], 'get'):
            interaction_matrix = sp.sparse.coo_matrix((
                weights,
                (
                    df[user_col].map(self.mapping['users_mapping'].get),
                    df[item_col].map(self.mapping['items_mapping'].get)
                )
            ))
        else:
            raise AttributeError

        self.users_watched = df.groupby(user_col).agg({item_col: list})
        return interaction_matrix

    def fit(self, train: pd.DataFrame):
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(train).tocsr().T

        self.dist_model.fit(
            self.weights_matrix,
            show_progress=(self.verbose > 0)
        )
        self.is_fitted = True

    @staticmethod
    def _generate_recs_mapper(
        model: ItemItemRecommender,
        user_mapping: Dict[int, int],
        user_inv_mapping: Dict[int, int],
        n_neighbors: int
    ):
        def _recs_mapper(user):
            user_id = user_mapping[user]
            recs = model.similar_items(user_id, N=n_neighbors)
            return (
                [user_inv_mapping[user] for user, _ in zip(*recs)],
                [sim for _, sim in zip(*recs)]
            )

        return _recs_mapper

    def predict(self, user_id: int, n_recs: int = 10):

        if not self.is_fitted:
            raise ValueError("Fit model before predicting")

        mapper = self._generate_recs_mapper(
            model=self.dist_model,
            user_mapping=self.mapping['users_mapping'],
            user_inv_mapping=self.mapping['users_inv_mapping'],
            n_neighbors=self.n_neighbors
        )

        recs = pd.DataFrame({'user_id': [user_id]})

        try:
            recs['sim_user_id'], recs['sim'] = zip(
                *recs['user_id'].map(mapper)
            )
        except AttributeError:
            return []

        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        recs = (
            recs
            .merge(
                self.users_watched,
                left_on=['sim_user_id'],
                right_on=['user_id'], how='left'
            )
            .explode('item_id')
            .sort_values(['user_id', 'sim'], ascending=False)
            .drop_duplicates(['user_id', 'item_id'], keep='first')
        )

        recs['rank'] = recs.groupby('user_id').cumcount() + 1

        result = recs[recs['rank'] <= n_recs][['user_id', 'item_id', 'rank']]

        return result.item_id.tolist()[:n_recs]
