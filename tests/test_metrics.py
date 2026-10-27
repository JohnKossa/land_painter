import numpy as np
import pytest

from land_painter.data.config import ROLE_SALES, ColumnRef, FileSpec, SalesMapping, parcel_only_config
from land_painter.data.dataset import build_dataset
from land_painter.data.sources import inspect
from land_painter.land_table import LOT, FlatRate, LandTable
from land_painter.metrics import compute, cod, cov, over_price_sales, ratio_study, value_range
from land_painter.neighborhoods import ROOT
from land_painter.sales import DISQUALIFIED
from land_painter.session import Session

from test_core import synthetic


def test_ratio_study_known_values():
    study = ratio_study(np.array([90.0, 100.0, 110.0]), np.array([100.0, 100.0, 100.0]))
    assert study.n == 3 and study.median == pytest.approx(1.0)
    assert study.cod == pytest.approx(100 * (0.1 + 0 + 0.1) / 3)
    assert study.prd == pytest.approx(1.0)
    x = np.log2([95.0, 100.0, 105.0])
    assert study.prb == pytest.approx(np.polyfit(x, [-0.1, 0.0, 0.1], 1)[0])


def test_ratio_study_regressive_prd():
    # Low-priced sale over-assessed, high-priced under-assessed -> PRD > 1, PRB < 0
    study = ratio_study(np.array([60.0, 800.0]), np.array([50.0, 1000.0]))
    assert study.prd > 1 and study.prb < 0


def test_undefined_only_when_impossible():
    empty = ratio_study(np.zeros(0), np.zeros(0))
    assert (empty.n, empty.median, empty.cod, empty.prd, empty.prb) == (0, None, None, None, None)
    single = ratio_study(np.array([50.0]), np.array([100.0]))
    assert single.median == 0.5 and single.cod == 0.0 and single.prd == pytest.approx(1.0)
    assert single.prb is None  # a regression needs two sales
    zero = ratio_study(np.array([0.0, 0.0]), np.array([100.0, 200.0]))
    assert zero.median == 0 and zero.cod is None and zero.prd is None and zero.prb is None
    assert cov(np.array([5.0])) is None and cod(np.zeros(0)) is None
    assert cov(np.array([1.0, 3.0])) == pytest.approx(100 * np.std([1, 3], ddof=1) / 2)
    r = value_range(np.array([3.0, 1.0, 2.0]))
    assert (r.n, r.minimum, r.mean, r.maximum) == (3, 1.0, 2.0, 3.0)


@pytest.fixture
def session(tmp_path):
    gdf = synthetic.make_parcels(6)
    gdf["area"] = [100.0, 200.0, 100.0, 400.0, 100.0, 0.0]
    gdf["frontage"] = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    gdf["cost"] = [0.0, 0.0, 50.0, 100.0, 300.0, 0.0]
    parcels = tmp_path / "parcels.parquet"
    gdf.to_parquet(parcels)
    sales = tmp_path / "sales.csv"
    sales.write_text("key,price,date,vacant\n"
                     "P0,1000,2024-01-01,Y\n"   # vacant: 10/area, 100/frontage
                     "P1,2400,2024-01-02,Y\n"   # vacant: 12/area, 240/frontage
                     "P2,250,2024-01-03,N\n"    # improved: residual 200
                     "P3,600,2024-01-04,N\n"    # improved: residual 500
                     "P4,200,2024-01-05,N\n")   # improved: land value will exceed price
    config = parcel_only_config(inspect(parcels))
    p = config.parcel_file
    config.files.append(FileSpec(str(sales), [ROLE_SALES]))
    config.sales = SalesMapping("key", "price", "date", "vacant", ["Y"])
    config.units = {"Area": ColumnRef(p, "area"), "Road frontage": ColumnRef(p, "frontage")}
    config.depreciated_cost = [ColumnRef(p, "cost")]
    return Session.new(build_dataset(config))


def everything(session):
    return np.ones(session.layer.n_parcels, dtype=bool)


def test_metrics_without_land_table(session):
    m = compute(session, everything(session))
    assert (m.n_parcels, m.n_vacant, m.n_improved) == (6, 2, 3)
    assert m.ratio.n == 0 and m.ratio.median is None  # no land values yet
    area = next(e for e in m.evidence if e.unit == "Area")
    assert (area.vacant.minimum, area.vacant.mean, area.vacant.maximum) == (10.0, 11.0, 12.0)
    # residuals per sqft: (250-50)/100 = 2, (600-100)/400 = 1.25, (200-300)/100 = -1
    assert (area.residual.minimum, area.residual.maximum) == (pytest.approx(-1.0), pytest.approx(2.0))
    frontage = next(e for e in m.evidence if e.unit == "Road frontage")
    assert frontage.cov > area.cov  # 100 vs 240 per foot is far more spread than 10 vs 12 per sqft
    assert m.best_unit == "Area"
    assert next(e for e in m.evidence if e.unit == LOT).cov == pytest.approx(frontage.cov)  # same frontage everywhere
    assert m.allocation_sales == pytest.approx(np.median([1 - 50 / 250, 1 - 100 / 600, 1 - 300 / 200]))
    assert m.allocation_table is None  # needs land values


def test_metrics_with_land_table(session):
    session.set_land_table(ROOT, LandTable("Area", FlatRate(10.0)))
    m = compute(session, everything(session), "Area")
    # ratio study on vacant sales: 1000/1000 and 2000/2400
    assert m.ratio.n == 2 and m.ratio.median == pytest.approx(np.median([1.0, 2000 / 2400]))
    assert m.spread_cod == 0.0  # value per sqft is 10 everywhere (parcel with 0 area is skipped)
    # built parcels 2, 3, 4: land 1000, 4000, 1000 vs cost 50, 100, 300
    assert m.allocation_table == pytest.approx(np.median([1000 / 1050, 4000 / 4100, 1000 / 1300]))
    # vacant-sales rate: mean(10, 12) = 11 per sqft
    assert m.allocation_vacant == pytest.approx(np.median([1100 / 1150, 4400 / 4500, 1100 / 1400]))
    # land value > price: P2 (1000 > 250), P3 (4000 > 600), P4 (1000 > 200)
    assert (m.over_price, m.over_price_checked) == (3, 5)
    assert over_price_sales(session, everything(session)).tolist() == [2, 3, 4]


def test_disqualified_sales_are_excluded(session):
    session.set_land_table(ROOT, LandTable("Area", FlatRate(10.0)))
    session.set_sale_flags([1, 4], DISQUALIFIED)
    m = compute(session, everything(session), "Area")
    assert (m.n_vacant, m.n_improved, m.ratio.n) == (1, 2, 1)
    assert m.ratio.prb is None  # one sale left
    assert m.over_price == 2


def test_scope_is_the_given_parcels(session):
    session.set_land_table(ROOT, LandTable(LOT, FlatRate(500.0)))
    mask = np.zeros(6, dtype=bool)
    mask[[0, 2]] = True
    m = compute(session, mask, LOT)
    assert (m.n_parcels, m.n_vacant, m.n_improved) == (2, 1, 1)
    assert m.ratio.median == pytest.approx(0.5)
