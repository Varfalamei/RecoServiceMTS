from starlette.testclient import TestClient

from service.api.models_zoo import KNNModelWithTop
from service.settings import ServiceConfig


def test_base_funcs(
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    user_id = 176549
    k_recs = 10
    model = KNNModelWithTop(path_to_reco="data/KNNBM25withAddFeatures.csv.gz")
    reco = model.reco_predict(user_id=user_id, k_recs=k_recs)
    assert len(reco) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in reco)
