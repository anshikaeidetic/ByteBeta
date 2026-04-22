from byte.benchmarking.contracts import BenchmarkItem, OutputContract
from byte.benchmarking.scoring import canonical_output, score_output


def _item(contract: OutputContract, expected, **kwargs) -> object:
    return BenchmarkItem(
        item_id="x",
        provider_track="deepseek",
        family="f",
        scenario="s",
        seed_id="seed",
        variant_id="v1",
        input_payload={"messages": [{"role": "user", "content": "x"}]},
        output_contract=contract,
        expected_value=expected,
        **kwargs,
    )


def test_scores_numeric_tolerance() -> None:
    item = _item(OutputContract.NUMERIC_TOLERANCE, "54.81%", tolerance=0.25)
    assert score_output(item, "54.7%")
    assert not score_output(item, "53.8%")


def test_scores_exact_and_enum_contracts() -> None:
    assert score_output(_item(OutputContract.EXACT_TEXT, "Paris"), "paris")
    assert score_output(_item(OutputContract.ENUM_LABEL, "ALLOW"), "allow")
    assert not score_output(_item(OutputContract.WORKFLOW_ACTION, "BLOCK"), "review")


def test_scores_json_schema_and_canonicalizes() -> None:
    item = _item(
        OutputContract.JSON_SCHEMA,
        {
            "type": "object",
            "required": ["ticket", "service"],
            "properties": {"ticket": {"type": "string"}, "service": {"type": "string"}},
        },
    )
    assert score_output(item, '{"service":"svc","ticket":"T-100"}')
    assert canonical_output(item, '{"service":"svc","ticket":"T-100"}') == '{"service":"svc","ticket":"T-100"}'


def test_scores_fallback_expected() -> None:
    item = _item(OutputContract.FALLBACK_EXPECTED, True)
    assert score_output(item, "", fallback_taken=True)
    assert not score_output(item, "", fallback_taken=False)
