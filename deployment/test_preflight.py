from deployment.preflight import configuration_errors, metadata_errors


def test_production_rejects_unbound_identity_and_placeholder_auth():
    failures = configuration_errors({"API_AUTH_ENABLED": "true", "API_KEY": "replace_in_production"}, True)
    assert any("GRAPHRAG_BUILD_ID" in error for error in failures)
    assert any("API_KEY" in error for error in failures)


def test_valid_production_configuration_and_https_origin():
    env = {"GRAPHRAG_BUILD_ID": "build_0123456789abcdef", "GRAPH_EMBEDDING_BACKEND": "chroma_onnx",
           "GRAPH_VECTOR_COLLECTION": "isolated_build_0123456789abcdef", "API_AUTH_ENABLED": "true",
           "API_KEY": "a" * 40, "CORS_ORIGINS": "https://demo.example.org", "QUERY_CACHE_TTL_SECONDS": "0"}
    assert configuration_errors(env, True) == []
    for invalid in ("*", "http://demo.example.org", "https://demo.example.org/private"):
        assert configuration_errors({**env, "CORS_ORIGINS": invalid}, True)


def test_vector_provenance_counts_wrong_build_hash_and_page_separately():
    pdfs = {"2025-10-K.pdf": {"sha256": "expected"}}
    good = {"build_id": "build", "source_filing": "2025-10-K.pdf", "document_sha256": "expected", "page": 80}
    assert metadata_errors([good], "build", pdfs, 1)["foreign_or_unbound_build"] == 0
    vector = {key: value for key, value in good.items() if key != "document_sha256"}
    vector["pdf_sha256"] = "expected"
    assert metadata_errors([vector], "build", pdfs, 1)["source_hash_mismatch"] == 0
    assert metadata_errors([{**vector, "document_sha256": "conflicting"}], "build", pdfs, 1)["source_hash_mismatch"] == 1
    bad = {**good, "build_id": None, "document_sha256": "other", "page": 0}
    report = metadata_errors([bad], "build", pdfs, 1)
    assert report["foreign_or_unbound_build"] == report["source_hash_mismatch"] == report["missing_page"] == 1
