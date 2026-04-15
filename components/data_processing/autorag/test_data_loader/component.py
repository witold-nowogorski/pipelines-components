from json import JSONDecodeError

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def test_data_loader(
    test_data_bucket_name: str,
    test_data_path: str,
    sample_size: int = 25,
    test_data: dsl.Output[dsl.Artifact] = None,
):
    """Download test data json file from S3 into a KFP artifact.

    The component reads S3-compatible credentials from environment variables
    (injected by the pipeline from a Kubernetes secret) and downloads a JSON
    test data file from the provided bucket and path to the output artifact.
    If the JSON is a list with more than ``sample_size`` entries, a deterministic
    random sample of ``sample_size`` questions is written to the artifact instead
    of the full file.

    Args:
        test_data_bucket_name: S3 (or compatible) bucket that contains the test
            data file.
        test_data_path: S3 object key to the JSON test data file.
        sample_size: Number of questions to keep from the benchmark. Set to 0 or
            a negative value to disable sampling. Defaults to 25.
        test_data: Output artifact that receives the downloaded (and optionally
            sampled) file.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.

    Raises:
        ValueError: If S3 credentials are missing or misconfigured.
        Exception: If the download fails or the path is not a JSON file.
    """
    import json
    import logging
    import os
    import random
    import sys

    import boto3
    from botocore.exceptions import ClientError, SSLError

    logger = logging.getLogger("Test Data Loader component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    if not test_data_bucket_name:
        raise TypeError("test_data_bucket_name must be a non-empty string")

    def get_test_data_s3():
        """Validate S3 credentials and download the JSON test data file."""

        class TestDataLoaderException(Exception):
            pass

        s3_creds = {k: os.environ.get(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]}
        for k, v in s3_creds.items():
            if v is None:
                raise ValueError(
                    "%s environment variable not set. Check if kubernetes secret was configured properly" % k
                )
        s3_creds["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION")

        def _make_s3_client(verify=True):
            return boto3.client(
                "s3",
                endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
                region_name=s3_creds["AWS_DEFAULT_REGION"],
                aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
                verify=verify,
            )

        s3_client = _make_s3_client()

        logger.info(f"Fetching test data from S3: bucket={test_data_bucket_name}, path={test_data_path}")
        try:
            logger.info(f"Starting download to {test_data.path}")
            s3_client.download_file(test_data_bucket_name, test_data_path, test_data.path)
            logger.info("Download completed successfully")
        except SSLError:
            logger.warning(
                "SSL error when downloading %s, retrying with verify=False",
                test_data_path,
            )
            s3_client = _make_s3_client(verify=False)
            s3_client.download_file(test_data_bucket_name, test_data_path, test_data.path)
            logger.info("Download completed successfully with verify=False")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                raise FileNotFoundError(
                    "Test data object not found in S3. bucket=%r, key=%r. "
                    "Check that test_data_key (pipeline parameter) is the full object key to an existing JSON file."
                    % (test_data_bucket_name, test_data_path)
                ) from e
            else:
                raise TestDataLoaderException("Failed to fetch %s: %s", test_data_path, e) from e
        except Exception as e:
            raise TestDataLoaderException("Failed to fetch %s: %s", test_data_path, e) from e

        try:
            with open(test_data.path, "r") as f:
                benchmark = json.load(f)
        except JSONDecodeError as e:
            raise TestDataLoaderException("test_data_path must point to a valid JSON file.") from e

        sample_seed = 10
        if sample_size and sample_size > 0 and isinstance(benchmark, list) and len(benchmark) > sample_size:
            logger.info(
                "Sampling %d of %d benchmark questions (seed=%d)",
                sample_size,
                len(benchmark),
                sample_seed,
            )
            rng = random.Random(sample_seed)
            sampled = rng.sample(benchmark, sample_size)
            with open(test_data.path, "w") as f:
                json.dump(sampled, f)

    get_test_data_s3()


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        test_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
