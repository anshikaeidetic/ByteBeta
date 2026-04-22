import logging
import os
import uuid
from typing import Any

from byte.manager.object_data.base import ObjectBase
from byte.utils import lazy_optional_module
from byte.utils.error import ByteErrorCode
from byte.utils.log import byte_log, log_byte_error

boto3 = lazy_optional_module("boto3", package="boto3")


class S3Storage(ObjectBase):
    """S3 storage"""

    def __init__(
        self, bucket: str, path_prefix: str, access_key: str, secret_key: str, endpoint: str | None = None
    ) -> None:
        self._session = boto3.Session(
            aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
        self._s3 = self._session.resource("s3")
        self._bucket = bucket
        self._path_prefix = path_prefix
        self._endpoint = endpoint

    def put(self, obj: Any) -> str:
        f_path = os.path.join(self._path_prefix, str(uuid.uuid4()))
        self._s3.Bucket(self._bucket).put_object(Key=str(f_path), Body=obj)
        return f_path

    def get(self, obj: str) -> Any:
        try:
            return self._s3.Bucket(self._bucket).Object(obj).get()["Body"].read()
        except Exception as exc:  # pragma: no cover - boto boundary
            log_byte_error(
                byte_log,
                logging.WARNING,
                "S3 object read failed.",
                error=exc,
                code=ByteErrorCode.STORAGE_READ,
                boundary="storage.read",
                stage="s3_object_get",
            )
            byte_log.error("obj:%s not exist", obj)
            return None

    def get_access_link(self, obj: str, expires: int = 3600) -> str:
        s3 = self._session.client("s3")
        link = s3.generate_presigned_url(
            ClientMethod="get_object",
            ExpiresIn=expires,
            Params={"Bucket": self._bucket, "Key": obj},
        )
        if self._endpoint:
            link = link.replace("s3.amazonaws.com/" + self._bucket, self._endpoint)
        return link

    def delete(self, to_delete: list[str]) -> None:
        self._s3.Bucket(self._bucket).delete_objects(
            Delete={"Objects": [{"Key": k} for k in to_delete]}
        )
