from typing import List, Optional, Sequence

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from pydantic import BaseModel

from service.api.exceptions import (
    CredentialError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.log import app_logger

from .config import config_env
from .models_zoo import DumpModel


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class NotFoundError(BaseModel):
    error_key: str
    error_message: str = "NotFound"
    error_loc: Optional[Sequence[str]]


class UnauthorizedError(BaseModel):
    error_key: str
    error_message: str = "Unauthorized"
    error_loc: Optional[Sequence[str]]


router = APIRouter()

api_query = APIKeyQuery(name=config_env["API_KEY_NAME"], auto_error=False)
api_header = APIKeyHeader(name=config_env["API_KEY_NAME"], auto_error=False)
token_bearer = HTTPBearer(auto_error=False)
models_zoo = {"model_1": DumpModel()}


async def get_api_key(
    api_key_query: str = Security(api_query),
    api_key_header: str = Security(api_header),
    token: HTTPAuthorizationCredentials = Security(token_bearer),
):
    if api_key_query == config_env["API_KEY"]:
        return api_key_query
    elif api_key_header == config_env["API_KEY"]:
        return api_key_header
    elif token is not None and token.credentials == config_env["API_KEY"]:
        return token.credentials
    raise CredentialError()


@router.get(path="/health", tags=["Health"])
async def health(api_key: APIKey = Depends(get_api_key)) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        404: {"model": NotFoundError, "user": NotFoundError},
        401: {"model": UnauthorizedError},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: APIKey = Depends(get_api_key),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    if model_name not in models_zoo.keys():
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    else:
        reco = models_zoo[model_name].reco_predict(user_id=user_id, k_recs=k_recs)

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
