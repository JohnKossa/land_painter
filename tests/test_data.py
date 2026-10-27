import polars as pl
import pytest

from land_painter.data.config import (
    CATEGORICAL, CONTINUOUS, DEPRECIATED_COST, FILL_MEDIAN, FILL_VALUE, ROLE_ATTRIBUTES, ROLE_PARCELS,
    ROLE_SALES, SALE_PRICE, Cleaning, ColumnRef, FileSpec, ImportConfig, SalesMapping,
    suggest, suggest_roles, unit_field,
)
from land_painter.data.dataset import build_dataset
from land_painter.data.review import (
    ERROR, DataError, clean_numeric, numeric_stats, prepare, profile_characteristics, to_date, to_number,
    vacant_flags,
)
from land_painter.data.sources import inspect

from test_core import synthetic


@pytest.fixture
def files(tmp_path):
    """A parcel file with attributes, a separate cost CSV, and a sales CSV with text-typed values."""
    gdf = synthetic.make_parcels(20)
    gdf["area"] = [1000.0] * 18 + [-2147483648.0, 0.0]
    gdf["zone"] = ["R1", "R2"] * 10
    gdf["tract"] = ["0101", "0102"] * 10
    gdf["owner"] = [f"Owner {i}" for i in range(20)]
    parcels = tmp_path / "parcels.parquet"
    gdf.to_parquet(parcels)

    costs = tmp_path / "costs.csv"
    rows = ["pid,bldg_cost,shed_cost"] + [f"P{i},{100 * i},{'' if i % 2 else 10}" for i in range(19)]
    rows += ["P0,999,999", "ZZZ,1,1"]  # a duplicate key and an unknown key
    costs.write_text("\n".join(rows))

    sales = tmp_path / "sales.csv"
    sales.write_text("\n".join([
        "key,price,date,vacant",
        'P1,"$1,000",2024-01-05,Y',
        "P2,2000,2024-02-05,N",
        "P3,abc,2024-03-05,N",
        "P2,3000,not a date,",
        "NOPE,4000,2024-04-05,Y",
    ]))
    return {"parcels": str(parcels), "costs": str(costs), "sales": str(sales)}


@pytest.fixture
def config(files):
    p, c = files["parcels"], files["costs"]
    return ImportConfig(
        files=[FileSpec(p, [ROLE_PARCELS]), FileSpec(c, [ROLE_ATTRIBUTES]), FileSpec(files["sales"], [ROLE_SALES])],
        parcel_key="key",
        attribute_keys={c: "pid"},
        sales=SalesMapping("key", "price", "date", "vacant", ["Y"]),
        depreciated_cost=[ColumnRef(c, "bldg_cost"), ColumnRef(c, "shed_cost")],
        units={"Area": ColumnRef(p, "area")},
        neighborhood=ColumnRef(p, "zone"),
    )


def test_inspect_and_suggestions(files):
    parcels, sales = inspect(files["parcels"]), inspect(files["sales"])
    assert parcels.geometry_column == "geometry" and "geometry" not in parcels.columns
    assert parcels.n_rows == 20 and sales.n_rows == 5
    assert suggest_roles(parcels) == [ROLE_PARCELS]
    assert suggest_roles(sales) == [ROLE_SALES]
    assert suggest("key", parcels.columns) == "key"
    assert suggest(unit_field("Area"), parcels.columns) == "area"
    assert suggest(unit_field("Lake frontage"), parcels.columns) is None


def test_config_round_trip_and_relocate(config, files):
    assert ImportConfig.from_json(config.to_json()) == config
    config.characteristic_kinds[f"{files['costs']}::bldg_cost"] = CONTINUOUS
    config.relocate(files["costs"], "new.csv")
    assert config.attribute_keys == {"new.csv": "pid"}
    assert {r.file for r in config.depreciated_cost} == {"new.csv"}
    assert list(config.characteristic_kinds) == ["new.csv::bldg_cost"]


def test_conversions():
    values, lost = to_number(pl.Series(["1,200", "$5", "x", None]))
    assert values.to_list() == [1200.0, 5.0, None, None] and lost == 1
    dates, lost = to_date(pl.Series(["01/02/2024", "03/04/2024", "bad"]))
    assert dates[0].month == 1 and dates[1].day == 4 and lost == 1
    assert vacant_flags(pl.Series([True, False, None]), ["true"]).to_list() == [True, False, None]
    assert vacant_flags(pl.Series(["y", "N", None]), ["Y"]).to_list() == [True, False, None]


def test_ambiguous_dates_are_reported_not_guessed(tmp_path, config, files):
    from land_painter.data.review import parse_dates

    parsed = parse_dates(pl.Series(["01/02/2024", "03/04/2024"]))
    assert parsed.format == "%m/%d/%Y" and "%d/%m/%Y" in parsed.alternatives
    assert parse_dates(pl.Series(["01/02/2024", "25/04/2024"])).format == "%d/%m/%Y"  # 25 can't be a month
    assert parse_dates(pl.Series(["2024-01-02"])).alternatives == []

    with open(files["sales"], "w") as f:
        f.write("key,price,date,vacant\nP1,100,01/02/2024,Y\nP2,200,03/04/2024,N\n")
    assert any("ambiguous" in i.message for i in prepare(config).issues)
    config.sales.date_format = "%d/%m/%Y"  # the user's choice silences the warning
    prepared = prepare(config)
    assert not any("ambiguous" in i.message for i in prepared.issues)
    assert prepared.sales["date"][0].month == 2


def test_numeric_stats_and_cleaning():
    values = pl.Series([10.0, 12.0, 11.0, 9.0, 10.0, 1e9, 0.0, -5.0, -2147483648.0, None])
    stats = numeric_stats(values)
    assert (stats.zero, stats.negative, stats.placeholder, stats.outlier, stats.missing) == (1, 1, 1, 1, 1)
    assert stats.minimum == -5.0 and stats.maximum == 1e9  # placeholder excluded from min

    cleaned = clean_numeric(values, stats, Cleaning(missing=["placeholder", "outlier"]))
    assert cleaned.null_count() == 3 and cleaned.max() == 12.0
    cleaned = clean_numeric(values, stats, Cleaning(missing=["zero", "negative"], fill=FILL_VALUE, value=7))
    assert cleaned.to_list()[6:] == [7.0, 7.0, -2147483648.0, 7.0]
    assert clean_numeric(values, stats, Cleaning()).equals(values)  # nothing changes unless chosen


def test_prepare_reports_key_problems(config):
    prepared = prepare(config)
    messages = " | ".join(i.message for i in prepared.issues)
    assert "costs.csv: 1 row repeat a key" in messages
    assert "costs.csv: 1 key not found" in messages
    assert "costs.csv: 1 parcel have no row" in messages
    assert "1 sale have keys not found" in messages
    assert "1 sale have no vacant/improved value" in messages
    assert not prepared.blocking

    costs = prepared.parcels[DEPRECIATED_COST].to_list()
    assert costs[:3] == [10.0, 100.0, 210.0]  # summed; missing shed cost ignored; first duplicate used
    assert costs[19] is None  # P19 has no cost row
    assert prepared.unparseable[SALE_PRICE] == 1 and prepared.unparseable["sale_date"] == 1
    assert prepared.sales["price"].to_list()[:3] == [1000.0, 2000.0, None]


def test_duplicate_parcel_keys_block(config, files):
    gdf = synthetic.make_parcels(5)
    gdf.loc[1, "key"] = gdf.loc[0, "key"]
    gdf.to_parquet(files["parcels"])
    config.neighborhood = None
    config.units = {}
    prepared = prepare(config)
    assert [i.severity for i in prepared.blocking] == [ERROR]
    with pytest.raises(DataError):
        build_dataset(config, prepared)


def test_build_dataset_applies_cleaning_and_aligns(config):
    config.cleaning[unit_field("Area")] = Cleaning(missing=["placeholder", "zero"], fill=FILL_MEDIAN)
    dataset = build_dataset(config)
    assert dataset.layer.n_parcels == 20
    assert dataset.parcels["key"].to_list() == list(dataset.layer.keys)
    assert dataset.parcels[unit_field("Area")].to_list()[-2:] == [1000.0, 1000.0]
    assert dataset.sales["key"].to_list() == ["P1", "P2", "P3", "P2"]  # unknown key dropped
    assert dataset.sales["parcel"].to_list() == [1, 2, 3, 2]
    assert dataset.neighborhood_codes().to_list()[:2] == ["R1", "R2"]


def test_characteristic_profiles(config, files):
    infos = {path: inspect(path) for path in config.parcel_level_files()}
    profiles = {(p.ref.file, p.ref.column): p for p in profile_characteristics(config, infos)}
    parcels = files["parcels"]
    assert (parcels, "key") not in profiles and (files["costs"], "pid") not in profiles
    assert profiles[(parcels, "zone")].suggested == CATEGORICAL and profiles[(parcels, "zone")].review is None
    assert profiles[(parcels, "tract")].suggested == CATEGORICAL and "numbers" in profiles[(parcels, "tract")].review
    assert profiles[(parcels, "area")].review.startswith("Only 3 distinct")
    assert profiles[(files["costs"], "bldg_cost")].suggested == CONTINUOUS
