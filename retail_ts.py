"""Регуляризация и диагностика рядов продаж из журнала Online Retail."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from warnings import catch_warnings, simplefilter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, median_abs_deviation
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss


DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DATA_SHA256 = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
REQUIRED_FIELDS = ["StockCode", "InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"]
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def load_transactions(path="data/Online Retail.xlsx"):
    """Загрузить официальный XLSX и проверить версию исходного файла."""
    path = Path(path)
    if path.exists():
        workbook = path.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as zipped:
            workbook = zipped.read("Online Retail.xlsx")

    if sha256(workbook).hexdigest() != DATA_SHA256:
        raise ValueError(f"Контрольная сумма {path.name} отличается от версии UCI.")

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(workbook)

    rows = pd.read_excel(
        BytesIO(workbook), dtype={"StockCode": "string", "InvoiceNo": "string"}
    )
    rows["InvoiceDate"] = pd.to_datetime(rows["InvoiceDate"], errors="coerce")
    for column in ["Quantity", "UnitPrice"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def transaction_flags(rows):
    """Вернуть маски для аудита. Причины исключения могут пересекаться."""
    return pd.DataFrame(
        {
            "missing_required": rows[REQUIRED_FIELDS].isna().any(axis=1),
            "cancellation": rows["InvoiceNo"]
            .str.upper()
            .str.startswith("C")
            .fillna(False),
            "nonpositive_qty": rows["Quantity"].le(0),
            "nonpositive_price": rows["UnitPrice"].le(0),
            "duplicate_copy": rows.duplicated(),
        },
        index=rows.index,
    )


def select_sales(rows, flags):
    excluded = flags.drop(columns="duplicate_copy").any(axis=1)
    return rows.loc[~excluded]


def daily_sales(rows, sales, stock_code):
    """Сохранить весь календарь и различать два вида дней без продаж SKU."""
    dates = rows["InvoiceDate"].dt.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D", name="date")
    product = sales.loc[sales["StockCode"].eq(stock_code)]
    if product.empty:
        raise ValueError(f"Нет подходящих продаж для StockCode {stock_code!r}.")

    shop_days = sales["InvoiceDate"].dt.normalize()
    product_days = product["InvoiceDate"].dt.normalize()
    daily = pd.DataFrame(index=calendar)
    daily["shop_lines"] = shop_days.value_counts().reindex(calendar, fill_value=0)
    daily["units"] = (
        product.groupby(product_days)["Quantity"].sum().reindex(calendar, fill_value=0)
    )
    daily["shop_active"] = daily["shop_lines"].gt(0)
    daily["product_active"] = daily["units"].gt(0)
    daily["coverage_uncertain"] = ~daily["shop_active"]
    return daily


def aggregate_sales(daily, freq):
    """Суммировать объём и измерить фактическое покрытие каждого календарного периода."""
    if freq not in {"D", "W-SUN", "MS"}:
        raise ValueError(f"Неподдерживаемая частота: {freq!r}")
    aggregate = daily.resample(freq).agg(
        units=("units", "sum"),
        calendar_days=("units", "size"),
        shop_days=("shop_active", "sum"),
        product_days=("product_active", "sum"),
        uncertain_days=("coverage_uncertain", "sum"),
    )
    expected = {"D": 1, "W-SUN": 7, "MS": aggregate.index.days_in_month}[freq]
    aggregate["full_period"] = aggregate["calendar_days"].eq(expected)
    return aggregate


def contiguous_dates(mask):
    """Найти расположение календарных разрывов, объединяя соседние даты."""
    if not isinstance(mask.index, pd.DatetimeIndex):
        raise TypeError("Ожидается календарный индекс.")
    dates = mask.index[mask]
    groups = pd.Series(dates, index=dates).diff().ne(pd.Timedelta(days=1)).cumsum()
    return (
        pd.Series(dates, index=dates)
        .groupby(groups)
        .agg(["min", "max", "size"])
        .rename(columns={"min": "начало", "max": "конец", "size": "дней"})
        .reset_index(drop=True)
    )


def describe_sales(y):
    """Объём, вариация и доля нулей в календарных интервалах."""
    return pd.Series(
        {
            "периодов": y.size,
            "сумма": y.sum(),
            "среднее": y.mean(),
            "медиана": y.median(),
            "σ": y.std(),
            "CV": y.std() / y.mean(),
            "P95": y.quantile(0.95),
            "пик": y.max(),
            "нулевых, %": y.eq(0).mean() * 100,
        }
    )


def upper_iqr_flags(y, multiplier=1.5):
    """Односторонний порог Тьюки для необычно высоких объёмов."""
    q1, q3 = y.quantile([0.25, 0.75])
    threshold = q3 + multiplier * (q3 - q1)
    return y.gt(threshold), float(threshold)


def peak_invoice_summary(product, dates, freq="D"):
    """Оценить долю крупнейшего счёта в выбранных днях или неделях."""
    invoices = product.groupby(
        [pd.Grouper(key="InvoiceDate", freq=freq), "InvoiceNo"],
        observed=True,
    )["Quantity"].sum()
    summary = (
        invoices.groupby(level="InvoiceDate")
        .agg(total="sum", invoices="size", largest_invoice="max")
        .rename_axis("date")
        .reindex(dates)
    )
    summary["largest_share"] = summary["largest_invoice"] / summary["total"]
    return summary


def trailing_anomalies(y, window=10, min_history=8, cutoff=3.5):
    """Сравнить объём с медианой и MAD предыдущих недель на шкале log1p."""
    logged = np.log1p(y)
    history = logged.shift().rolling(window, min_periods=min_history)
    center = history.median()
    mad = history.apply(
        lambda values: median_abs_deviation(values, nan_policy="omit"), raw=True
    )
    scale = 1.4826 * mad.where(mad.gt(0))
    score = (logged - center) / scale
    return pd.DataFrame(
        {
            "baseline": np.expm1(center),
            "upper_limit": np.expm1(center + cutoff * scale),
            "score": score,
            "flag": score.gt(cutoff),
        }
    )


def weekday_profile(daily):
    """Сравнить календарный профиль и профиль по дням с продажами магазина."""
    day = daily.index.dayofweek
    active = daily.loc[daily["shop_active"], "units"]
    profile = daily.groupby(day).agg(
        days=("units", "size"),
        shop_days=("shop_active", "sum"),
        calendar_mean=("units", "mean"),
    )
    profile["active_mean"] = active.groupby(active.index.dayofweek).mean()
    profile["active_median"] = active.groupby(active.index.dayofweek).median()
    profile.index = pd.Index(WEEKDAYS, name="день")
    return profile


def weekday_sales_matrix(daily):
    """Построить матрицу неделя × день. Серии без покрытия получают NaN."""
    day = daily.index.dayofweek
    sunday = daily.index + pd.to_timedelta(6 - day, unit="D")
    return (
        daily["units"]
        .where(daily["shop_active"])
        .to_frame()
        .assign(week=sunday, weekday=day)
        .pivot(index="week", columns="weekday", values="units")
        .reindex(columns=range(7))
        .set_axis(WEEKDAYS, axis="columns")
    )


def weekday_permutation_test(samples, block_weeks=1, resamples=19_999, seed=21790):
    """Тест Фридмана с перестановками названий дней внутри блоков недель."""
    values = samples.to_numpy()
    n, k = values.shape
    if n < 2 or k < 3 or samples.isna().any().any():
        raise ValueError("Нужны сопоставимые недели без пропусков и минимум три дня.")
    if block_weeks < 1 or resamples < 1 or not samples.index.is_monotonic_increasing:
        raise ValueError("Нужны положительные размеры и хронологический порядок.")

    # Поправка Фридмана на одинаковые ранги постоянна при перестановках.
    ranks = samples.rank(axis=1).sub((k + 1) / 2)
    statistic = (
        friedmanchisquare(*values.T).statistic if ranks.ne(0).any().any() else 0.0
    )
    week_number = (samples.index - samples.index[0]).days // 7
    blocks = ranks.groupby(week_number // block_weeks).sum().to_numpy()
    if len(blocks) < 2:
        raise ValueError("Тесту нужны хотя бы два блока.")

    rng = np.random.default_rng(seed)
    permutations = rng.random((resamples, len(blocks), k)).argsort(axis=-1)
    simulated = np.take_along_axis(blocks[None, :, :], permutations, axis=-1)
    perm_scores = np.square(simulated.sum(axis=1)).sum(axis=1)
    observed_score = np.square(blocks.sum(axis=0)).sum()
    p_value = (np.count_nonzero(perm_scores >= observed_score) + 1) / (resamples + 1)
    return pd.Series(
        {
            "недель": n,
            "блоков": len(blocks),
            "χ² Фридмана": statistic,
            "W Кендалла": statistic / (n * (k - 1)),
            "p перестановок": p_value,
        }
    )


def decompose(y, period=7):
    """Робастное аддитивное STL и мера силы сезонной компоненты."""
    model = STL(y, period=period, robust=True).fit()
    remainder = model.resid + model.seasonal
    strength = (
        max(0.0, 1 - model.resid.var() / remainder.var())
        if remainder.var() > 0
        else 0.0
    )
    return model, strength


def autocorrelations(y, nlags=28):
    return pd.Series(acf(y, nlags=nlags), name="ACF").rename_axis("лаг")


def stationarity_tests(y, regression="c", kpss_lags="auto"):
    """ADF и KPSS с константой или константой и линейным трендом."""
    if y.isna().any() or y.nunique() < 2:
        raise ValueError("Тестам нужен ряд без пропусков с меняющимися значениями.")
    adf_stat, adf_p, adf_lags, _, _, _ = adfuller(
        y, regression=regression, autolag="AIC"
    )
    with catch_warnings(record=True) as caught:
        simplefilter("always", InterpolationWarning)
        kpss_stat, kpss_p, used_kpss_lags, _ = kpss(
            y, regression=regression, nlags=kpss_lags
        )
    p_bound = "="
    if any(issubclass(w.category, InterpolationWarning) for w in caught):
        p_bound = ">" if kpss_p == 0.10 else "<"
    shown_kpss_p = f"{kpss_p:.4g}" if p_bound == "=" else f"{p_bound} {kpss_p:.2f}"
    return pd.DataFrame(
        {
            "статистика": [adf_stat, kpss_stat],
            "p": [f"{adf_p:.4g}", shown_kpss_p],
            "лаги": [adf_lags, used_kpss_lags],
            "H₀ отклонена при 5%": [adf_p < 0.05, kpss_p < 0.05],
        },
        index=pd.Index(["ADF", "KPSS"], name="тест"),
    )
