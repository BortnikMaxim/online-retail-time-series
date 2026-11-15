"""Графики для анализа календарных продаж."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from retail_ts import autocorrelations


INK = "#172838"
MUTED = "#566879"
BLUE = "#205989"
TEAL = "#11736E"
ORANGE = "#AD4D24"
PALE = "#EEF3F8"
EMPTY = "#BBC5D0"


def set_style():
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": EMPTY,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "grid.color": "#E4EBF1",
            "grid.linewidth": 0.75,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "legend.fontsize": 9,
        },
    )


def date_axis(ax, interval=2):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    ax.grid(axis="x", visible=False)


def plot_overview(daily, weekly, monthly, longest_gap):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True, layout="constrained")
    fig.suptitle("Продажи на трёх частотах", x=0.05, ha="left", weight="bold")

    axes[0].plot(
        daily.index, daily["units"], color=BLUE, lw=1.0, label="Объём по журналу"
    )
    axes[0].set(title="По дням", ylabel="Штук в день")

    monthly_centers = monthly.index + pd.to_timedelta(
        monthly["calendar_days"].to_numpy() // 2, unit="D"
    )
    for ax, data, x, title, unit in [
        (axes[1], weekly, weekly.index, "По неделям", "Штук в неделю"),
        (axes[2], monthly, monthly_centers, "По месяцам", "Штук в месяц"),
    ]:
        full = data["full_period"]
        widths = data["calendar_days"] * 0.72
        ax.bar(
            x[full],
            data.loc[full, "units"],
            width=widths[full],
            color=BLUE,
            label="Полный календарный период",
        )
        if (~full).any():
            ax.bar(
                x[~full],
                data.loc[~full, "units"],
                width=widths[~full],
                color=EMPTY,
                label="Неполный крайний период",
            )
        empty = data["shop_days"].eq(0)
        if empty.any():
            ax.scatter(
                x[empty],
                data.loc[empty, "units"],
                color=ORANGE,
                marker="D",
                s=30,
                zorder=4,
                label="Период без записей магазина",
            )
        ax.set(title=title, ylabel=unit)

    for ax in axes:
        ax.axvspan(
            longest_gap["начало"],
            longest_gap["конец"] + pd.Timedelta(days=1),
            color=ORANGE,
            alpha=0.10,
        )
        ax.legend(loc="upper left")
        ax.set_ylim(bottom=0)
    axes[0].annotate(
        f"Дней без записей магазина: {longest_gap['дней']}",
        xy=(longest_gap["конец"], 80),
        xytext=(15, 31),
        textcoords="offset points",
        color=ORANGE,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": ORANGE},
    )
    date_axis(axes[-1])
    axes[-1].set_xlabel("Календарная дата")
    return fig


def plot_coverage(matrix):
    fig, ax = plt.subplots(figsize=(12, 3.65), layout="constrained")
    heat = np.log1p(matrix.T)
    cmap = sns.light_palette(TEAL, as_cmap=True)
    levels = np.array([0, 5, 20, 50, 100, 200, 400])
    ax.set_facecolor(EMPTY)
    sns.heatmap(
        heat,
        cmap=cmap,
        vmin=0,
        vmax=np.log1p(levels[-1]),
        linewidths=0.55,
        linecolor="white",
        xticklabels=False,
        cbar_kws={
            "ticks": np.log1p(levels),
            "pad": 0.02,
            "label": "Продано, шт. (шкала log1p)",
        },
        ax=ax,
    )
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(levels)
    month_start = ~matrix.index.to_period("M").duplicated()
    positions = np.flatnonzero(month_start)
    ax.set_xticks(
        positions + 0.5, matrix.index[positions].strftime("%m.%Y"), fontsize=9
    )
    for x in positions[1:]:
        ax.axvline(x, color="white", lw=1.4)
    ax.tick_params(axis="y", rotation=0)
    ax.set(
        title="Дневные продажи и покрытие журнала",
        xlabel="Месяц окончания недели · серые клетки — нет записей магазина",
        ylabel="День недели",
    )
    return fig


def plot_dynamics(weekly, rate):
    y = weekly["units"]
    rolling = y.rolling(8, min_periods=8)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.9), sharex=True, layout="constrained")
    fig.suptitle("Рост объёма и разброса по неделям", x=0.05, ha="left", weight="bold")

    axes[0].plot(
        y.index, y, color=BLUE, alpha=0.5, lw=1.15, label="Зафиксированный объём"
    )
    axes[0].plot(
        y.index, rolling.mean(), color=BLUE, lw=2.2, label="Среднее за 8 недель"
    )
    axes[0].set(title="Уровень продаж", ylabel="Штук в неделю")
    axes[0].legend(loc="upper left")

    sd = rolling.std()
    axes[1].fill_between(sd.index, 0, sd, color=ORANGE, alpha=0.16)
    axes[1].plot(sd.index, sd, color=ORANGE, lw=2)
    axes[1].set(title="Стандартное отклонение за 8 недель", ylabel="Штук")

    axes[2].plot(
        rate.index,
        rate,
        color=TEAL,
        alpha=0.55,
        lw=1.1,
        label="Недельный объём ÷ активные дни",
    )
    rolling_rate = rate.rolling(8, min_periods=8).median()
    axes[2].plot(
        rate.index, rolling_rate, color=TEAL, lw=2.2, label="Медиана за 8 недель"
    )
    short = weekly["shop_days"].lt(6)
    axes[2].scatter(
        rate.index[short],
        rate[short],
        marker="o",
        s=30,
        facecolor="white",
        edgecolor=TEAL,
        zorder=4,
        label="Недели с 4–5 активными днями",
    )
    axes[2].set(
        title="Объём в расчёте на активный день",
        xlabel="Конец недели",
        ylabel="Штук в день",
    )
    axes[2].legend(loc="upper left", ncols=2)
    date_axis(axes[-1])
    return fig


def plot_peaks(daily, day_flags, day_limit, scores, week_flags, score_limit=3.5):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.9), layout="constrained")
    fig.suptitle("Крупные продажи на фоне истории", x=0.05, ha="left", weight="bold")

    axes[0].plot(daily.index, daily, color=BLUE, lw=0.9)
    axes[0].scatter(
        daily.index[day_flags], daily[day_flags], color=ORANGE, s=19, zorder=3
    )
    axes[0].axhline(
        day_limit,
        color=ORANGE,
        linestyle="--",
        lw=1.0,
        label=f"Порог Тьюки {day_limit:.1f} шт.",
    )
    axes[0].set(
        title="Дневные пики по общему порогу", ylabel="Штук в день", xlabel="Дата"
    )
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(bottom=0)

    valid = scores.dropna(subset=["score"])
    global_flags = week_flags.reindex(valid.index)
    local_flags = valid["flag"]
    axes[1].plot(
        valid.index,
        valid["score"],
        color=BLUE,
        lw=1.15,
        label="Робастная оценка отклонения",
    )
    axes[1].axhline(0, lw=0.8, color=EMPTY)
    axes[1].axhline(
        score_limit,
        color=TEAL,
        lw=1,
        linestyle="--",
        label=f"Локальный порог {score_limit:g}",
    )
    axes[1].scatter(
        valid.index[global_flags],
        valid.loc[global_flags, "score"],
        c=ORANGE,
        s=48,
        marker="o",
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
        label="Помечено общим недельным порогом",
    )
    axes[1].scatter(
        valid.index[local_flags],
        valid.loc[local_flags, "score"],
        c=TEAL,
        s=62,
        marker="D",
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
        label="Помечено локальным порогом",
    )
    axes[1].set(
        title="Недельное отклонение от предшествующего уровня",
        xlabel="Конец недели",
        ylabel="Робастный z на шкале log1p",
    )
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncols=2)
    axes[1].margins(y=0.16)
    for ax in axes:
        date_axis(ax)
    return fig


def plot_weekdays(active):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.55),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    fig.suptitle("День недели в сопоставимых неделях", x=0.05, ha="left", weight="bold")
    long = active.melt(var_name="день", value_name="продажи")
    long["log_sales"] = np.log1p(long["продажи"])
    sns.boxplot(
        data=long,
        x="день",
        y="log_sales",
        order=list(active),
        ax=axes[0],
        color="#A6CFCA",
        width=0.55,
        fliersize=0,
        linewidth=1.15,
    )

    rng = np.random.default_rng(21790)
    for i, column in enumerate(active):
        coords = i + rng.uniform(-0.18, 0.18, size=len(active))
        axes[0].scatter(
            coords, np.log1p(active[column]), s=15, color=TEAL, alpha=0.37, linewidth=0
        )
    levels = np.array([0, 5, 20, 50, 100, 200, 400])
    axes[0].set_yticks(np.log1p(levels), levels)
    axes[0].set(
        title="Дневные объёмы",
        xlabel="День недели",
        ylabel="Продано, шт. (шкала log1p)",
    )

    ranks = active.rank(axis=1)
    midpoint = len(ranks) // 2
    x = np.arange(len(active.columns))
    axes[1].axhline(
        (len(active.columns) + 1) / 2,
        color=MUTED,
        ls="--",
        lw=1,
        label="Средний ранг при равенстве",
    )
    axes[1].plot(
        x,
        ranks.iloc[:midpoint].mean(),
        marker="o",
        color=BLUE,
        label="Первая половина",
    )
    axes[1].plot(
        x,
        ranks.iloc[midpoint:].mean(),
        marker="D",
        color=ORANGE,
        label="Вторая половина",
    )
    axes[1].set(
        xticks=x,
        xticklabels=list(active),
        title="Средний ранг в пределах недели",
        xlabel="День недели",
        ylabel="Ранг (от 1 до 6)",
    )
    axes[1].set_ylim(1.0, 5.45)
    axes[1].legend(loc="upper right")
    return fig


def plot_decomposition(y, stl):
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 8.25), layout="constrained", sharex=True
    )
    fig.suptitle("STL дневных продаж · 7 дней", x=0.05, ha="left", weight="bold")
    components = [
        ("Зафиксированный объём", y, BLUE),
        ("Тренд", stl.trend, TEAL),
        ("Сезонная компонента", stl.seasonal, BLUE),
        ("Остаток", stl.resid, ORANGE),
    ]
    for i, (ax, (title, series, color)) in enumerate(zip(axes, components)):
        ax.plot(series.index, series, color=color, lw=1.1)
        if i >= 2:
            ax.axhline(0, color=EMPTY, lw=0.85)
            ax.fill_between(series.index, 0, series, color=color, alpha=0.12)
        ax.set(title=title, ylabel="Штук")
    axes[-1].set_xlabel("Дата")
    date_axis(axes[-1])
    return fig


def plot_seasonal_acf(y, residual, nlags=28):
    fig, ax = plt.subplots(figsize=(12, 3.1), layout="constrained")
    for name, series, color in [
        ("Дневной ряд", y, BLUE),
        ("Остаток STL", residual, TEAL),
    ]:
        corr = autocorrelations(series, nlags=nlags).iloc[1:]
        ax.plot(corr.index, corr, color=color, lw=1.3, marker="o", ms=3.5, label=name)
    ax.axhline(0, color=EMPTY, lw=0.85)
    for lag in range(7, nlags + 1, 7):
        ax.axvline(lag, color=ORANGE, lw=0.85, alpha=0.3)
    ax.set(
        title="Связь на лагах, кратных семи",
        xlabel="Лаг, дней",
        ylabel="ACF",
        xticks=[1, 7, 14, 21, 28],
    )
    ax.legend(loc="upper right")
    return fig


def plot_transformation(y, difference):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 5.9),
        layout="constrained",
        sharex="col",
        gridspec_kw={"width_ratios": [1.85, 1]},
    )
    fig.suptitle(
        "Недельный ряд до и после преобразования", x=0.05, ha="left", weight="bold"
    )
    inputs = [
        (y, "Недельные продажи", "Штук в неделю", BLUE),
        (difference, "Первая разность log1p", "Разность log", TEAL),
    ]
    for i, (series, title, unit, color) in enumerate(inputs):
        ax = axes[i, 0]
        ax.plot(series.index, series, color=color, lw=1.35, marker="o", ms=3)
        ax.set(title=title, ylabel=unit)
        if i == 1:
            ax.axhline(0, color=EMPTY, lw=0.9)
        acf_ax = axes[i, 1]
        corr = autocorrelations(series, nlags=12).iloc[1:]
        acf_ax.axhline(0, color=EMPTY, lw=0.85)
        acf_ax.vlines(corr.index, 0, corr, color=color, lw=1.4)
        acf_ax.scatter(corr.index, corr, color=color, s=15)
        acf_ax.set(title="Автокорреляция", ylabel="ACF", ylim=(-0.62, 0.62))
    axes[-1, 0].set_xlabel("Конец недели")
    axes[-1, 1].set(xlabel="Лаг, недель", xticks=[1, 3, 6, 9, 12])
    date_axis(axes[-1, 0])
    return fig
