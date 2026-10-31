import pytest

from monte_carlo_accelerator.portfolio_io import load_term_life_portfolio


def test_load_term_life_portfolio(tmp_path):
    source = tmp_path / "portfolio.csv"
    source.write_text(
        "policy_id,age_at_entry,sum_assured,annual_premium,term_years\n"
        "101,35,250000,1250,20\n"
        "102,51,100000,900,10\n",
        encoding="utf-8",
    )

    policies = load_term_life_portfolio(source)

    assert len(policies) == 2
    assert policies.policy_id.tolist() == [101, 102]
    assert policies.sum_assured.tolist() == [250_000.0, 100_000.0]


@pytest.mark.parametrize(
    "body,message",
    [
        ("policy_id,age_at_entry\n1,30\n", "missing required columns"),
        (
            "policy_id,age_at_entry,sum_assured,annual_premium,term_years\n"
            "1,30,100000,500,10\n1,40,120000,700,8\n",
            "unique",
        ),
        (
            "policy_id,age_at_entry,sum_assured,annual_premium,term_years\n"
            "1,130,100000,500,10\n",
            "age_at_entry",
        ),
    ],
)
def test_invalid_portfolios_are_rejected(tmp_path, body, message):
    source = tmp_path / "portfolio.csv"
    source.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_term_life_portfolio(source)
