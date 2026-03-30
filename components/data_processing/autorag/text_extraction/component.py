from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=["docling[ort]"],
)
def text_extraction(
    documents_descriptor: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Output[dsl.Artifact],
):
    """Text Extraction component.

    Reads the documents_descriptor JSON (from documents_discovery), fetches
    the listed documents from S3, and extracts text using the docling library.

    Args:
        documents_descriptor: Input artifact containing
            documents_descriptor.json with bucket, prefix, and documents list.
        extracted_text: Output artifact where the extracted text content will be stored.
    """
    import json
    import logging
    import multiprocessing
    import os
    import sys
    import tempfile
    import threading
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from pathlib import Path

    import boto3
    from botocore.exceptions import SSLError

    DOCUMENTS_DESCRIPTOR_FILENAME = "documents_descriptor.json"
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}
    DOWNLOAD_MAX_WORKERS = 8

    logger = logging.getLogger("Text Extraction component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    descriptor_root = Path(documents_descriptor.path)
    if descriptor_root.is_dir():
        descriptor_path = descriptor_root / DOCUMENTS_DESCRIPTOR_FILENAME
    else:
        descriptor_path = descriptor_root

    if not descriptor_path.exists():
        raise FileNotFoundError(f"Descriptor not found: {descriptor_path}")

    with open(descriptor_path) as f:
        descriptor = json.load(f)

    bucket = descriptor["bucket"]
    documents = descriptor["documents"]

    s3_creds = {k: os.environ.get(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]}
    for k, v in s3_creds.items():
        if v is None:
            raise ValueError(f"{k} environment variable not set. Check if kubernetes secret was configured properly.")

    s3_creds["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "")

    def _make_s3_client(verify=True):
        session = boto3.session.Session(
            aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
            region_name=s3_creds.get("AWS_DEFAULT_REGION"),
        )
        return session.client(
            service_name="s3",
            endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
            verify=verify,
        )

    _thread_local = threading.local()

    def _get_thread_s3_client():
        if not hasattr(_thread_local, "s3_client"):
            _thread_local.s3_client = _make_s3_client()
        return _thread_local.s3_client

    def download_document(doc: dict, base_path: Path) -> Path:
        key = doc["key"]
        local_path = base_path / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Downloading %s", key)
            _get_thread_s3_client().download_file(bucket, key, str(local_path))
            return local_path
        except SSLError:
            logger.warning(
                "SSL error when downloading %s, retrying with verify=False",
                key,
            )
            _thread_local.s3_client = _make_s3_client(verify=False)
            _thread_local.s3_client.download_file(bucket, key, str(local_path))
            return local_path
        except Exception as e:
            logger.error("Failed to fetch %s: %s", key, e)
            raise

    def _proc_initializer() -> None:
        """Runs once per worker process: creates a single DocumentConverter."""
        import logging as _log_mod
        import sys as _sys_mod

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PaginatedPipelineOptions, PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            HTMLFormatOption,
            MarkdownFormatOption,
            PdfFormatOption,
            PowerpointFormatOption,
            WordFormatOption,
        )

        _wlog = _log_mod.getLogger("text_extraction_worker")
        _wlog.setLevel(_log_mod.INFO)
        if not _wlog.handlers:
            _wlog.addHandler(_log_mod.StreamHandler(_sys_mod.stdout))

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = False
        pdf_opts.do_table_structure = False
        pdf_opts.accelerator_options = AcceleratorOptions(device="cpu", num_threads=1)

        pag_opts = PaginatedPipelineOptions()
        pag_opts.generate_page_images = False
        pag_opts.accelerator_options = AcceleratorOptions(device="cpu", num_threads=1)

        _sys_mod.modules["__main__"]._worker_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_options=pag_opts),
                InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pag_opts),
                InputFormat.HTML: HTMLFormatOption(),
                InputFormat.MD: MarkdownFormatOption(),
            }
        )

    def _proc_process_document(file_path_str: str, output_dir_str: str) -> bool:
        """Process one document using the worker-process-local converter."""
        import logging as _log_mod
        import sys as _sys_mod
        from pathlib import Path as _Path

        _wlog = _log_mod.getLogger("text_extraction_worker")
        try:
            path = _Path(file_path_str)
            out_dir = _Path(output_dir_str)
            output_file = out_dir / f"{path.name}.md"

            if path.suffix.lower() == ".txt":
                _wlog.info("Processing TXT file %s (direct read)", path.name)
                output_file.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                return True

            converter = _sys_mod.modules["__main__"]._worker_converter
            _wlog.debug("Processing %s with docling", path.name)
            result = converter.convert(path)
            output_file.write_text(result.document.export_to_markdown(), encoding="utf-8")
            _wlog.debug("Successfully processed %s", path.name)
            return True
        except Exception as e:
            _log_mod.getLogger("text_extraction_worker").error("Failed to process %s: %s", file_path_str, e)
            return False

    _proc_initializer.__qualname__ = "_proc_initializer"
    _proc_initializer.__module__ = "__main__"
    _proc_process_document.__qualname__ = "_proc_process_document"
    _proc_process_document.__module__ = "__main__"
    _main_mod = sys.modules["__main__"]
    _main_mod._proc_initializer = _proc_initializer
    _main_mod._proc_process_document = _proc_process_document

    output_dir = Path(extracted_text.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not documents:
        logger.info("No documents to process.")
        return

    logger.info("Starting text extraction for %d documents.", len(documents))

    n_workers = os.cpu_count() or 1

    with tempfile.TemporaryDirectory() as download_dir:
        download_path = Path(download_dir)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_main_mod._proc_initializer,
        ) as process_pool:
            process_futures = []

            with ThreadPoolExecutor(max_workers=DOWNLOAD_MAX_WORKERS) as dl_pool:
                dl_futures = {dl_pool.submit(download_document, doc, download_path): doc for doc in documents}

                for dl_future in as_completed(dl_futures):
                    try:
                        local_path = dl_future.result()
                    except Exception:
                        continue

                    if local_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        pf = process_pool.submit(
                            _main_mod._proc_process_document,
                            str(local_path),
                            str(output_dir),
                        )
                        process_futures.append(pf)
                    else:
                        logger.warning("Unsupported format, skipping: %s", local_path)

            all_results = [f.result() for f in process_futures]

    processed_count = sum(1 for r in all_results if r)
    error_count = len(all_results) - processed_count
    logger.info("Text extraction completed. Total processed: %d, Errors: %d.", processed_count, error_count)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        text_extraction,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
