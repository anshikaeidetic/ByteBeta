import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import requests

from byte.manager import ObjectBase
from byte.manager.object_data import s3_storage as s3_module
from byte.manager.object_data.local_storage import LocalObjectStorage
from byte.manager.object_data.s3_storage import S3Storage


class TestLocal(unittest.TestCase):
    def test_normal(self) -> None:
        with TemporaryDirectory() as root:
            o = LocalObjectStorage(root)
            data = b"My test"
            fp = o.put(data)
            self.assertTrue(Path(fp).is_file())
            self.assertEqual(o.get(fp), data)
            self.assertEqual(o.get_access_link(fp), fp)
            o.delete([fp])
            self.assertFalse(Path(fp).is_file())

    def test_rejects_paths_outside_root(self) -> None:
        with TemporaryDirectory() as root:
            o = LocalObjectStorage(root)
            with TemporaryDirectory() as outside_root:
                outside_file = Path(outside_root) / "secret.bin"
                outside_file.write_bytes(b"secret")

                self.assertIsNone(o.get(str(outside_file)))
                o.delete([str(outside_file)])
                self.assertTrue(outside_file.is_file())


class TestS3(unittest.TestCase):
    def test_normal(self) -> None:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        bucket = os.environ.get("BUCKET")
        endpoint = os.environ.get("ENDPOINT")
        if access_key is None or secret_key is None or bucket is None:
            return
        o = S3Storage(bucket, "byte", access_key, secret_key, endpoint)
        data = b"My test"
        fp = o.put(data)
        self.assertEqual(o.get(fp), data)
        link = o.get_access_link(fp)
        self.assertEqual(requests.get(link, verify=False).content, data)
        o.delete([fp])
        self.assertIsNone(o.get(fp))

    def test_s3_storage_methods_with_mocked_session(self) -> None:
        body = mock.Mock()
        body.read.return_value = b"My test"
        bucket = mock.Mock()
        bucket.Object.return_value.get.return_value = {"Body": body}
        session = mock.Mock()
        session.resource.return_value.Bucket.return_value = bucket
        session.client.return_value.generate_presigned_url.return_value = (
            "https://s3.amazonaws.com/byte/path/object-id"
        )

        fake_boto3 = SimpleNamespace(Session=mock.Mock(return_value=session))
        with mock.patch.object(s3_module, "boto3", fake_boto3):
            with mock.patch.object(s3_module.uuid, "uuid4", return_value="object-id"):
                storage = S3Storage(
                    bucket="byte",
                    path_prefix="path",
                    access_key="access",
                    secret_key="secret",
                    endpoint="objects.byte.internal",
                )

                key = storage.put(b"My test")
                assert key == os.path.join("path", "object-id")
                assert storage.get(key) == b"My test"
                assert storage.get_access_link(key) == "https://objects.byte.internal/path/object-id"
                storage.delete([key])

        bucket.put_object.assert_called_once_with(Key=os.path.join("path", "object-id"), Body=b"My test")
        session.client.return_value.generate_presigned_url.assert_called_once()
        bucket.delete_objects.assert_called_once_with(
            Delete={"Objects": [{"Key": os.path.join("path", "object-id")}]}
        )

    def test_s3_storage_returns_none_when_read_fails(self) -> None:
        bucket = mock.Mock()
        bucket.Object.return_value.get.side_effect = RuntimeError("missing")
        session = mock.Mock()
        session.resource.return_value.Bucket.return_value = bucket

        fake_boto3 = SimpleNamespace(Session=mock.Mock(return_value=session))
        with mock.patch.object(s3_module, "boto3", fake_boto3):
            storage = S3Storage(
                bucket="byte",
                path_prefix="path",
                access_key="access",
                secret_key="secret",
            )

            assert storage.get("path/missing") is None


class TestBase(unittest.TestCase):
    def test_local(self) -> None:
        with TemporaryDirectory() as root:
            o = ObjectBase("local", path=root)
            data = b"My test"
            fp = o.put(data)
            self.assertTrue(Path(fp).is_file())
            self.assertEqual(o.get(fp), data)
            self.assertEqual(o.get_access_link(fp), fp)
            o.delete([fp])
            self.assertFalse(Path(fp).is_file())

    def test_s3(self) -> None:
        bucket = mock.Mock()
        bucket.Object.return_value.get.return_value = {"Body": mock.Mock(read=mock.Mock(return_value=b"My test"))}
        fake_session = mock.Mock()
        fake_session.resource.return_value.Bucket.return_value = bucket
        fake_session.client.return_value.generate_presigned_url.return_value = (
            "https://s3.amazonaws.com/byte-bucket/objects/object-id"
        )
        fake_boto3 = SimpleNamespace(Session=mock.Mock(return_value=fake_session))
        with mock.patch.object(s3_module, "boto3", fake_boto3):
            with mock.patch.object(s3_module.uuid, "uuid4", return_value="object-id"):
                o = ObjectBase(
                    "s3",
                    bucket="byte-bucket",
                    path_prefix="objects",
                    access_key="access",
                    secret_key="secret",
                )
                data = b"My test"
                fp = o.put(data)
                o.get(fp)
                o.get_access_link(fp)

        bucket.put_object.assert_called_once_with(Key=os.path.join("objects", "object-id"), Body=b"My test")
