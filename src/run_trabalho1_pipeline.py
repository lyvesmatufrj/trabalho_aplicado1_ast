from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
TABLE_DIR = BASE_DIR / "reports" / "tables"
LOG_DIR = BASE_DIR / "reports" / "logs"

SOURCE_CSV = BASE_DIR / "dataset_pmc_selic_monthly.csv"
RAW_CSV = RAW_DIR / "dataset_pmc_selic_monthly.csv"
RAW_PMC_CSV = RAW_DIR / "dataset_pmc_monthly.csv"
INTERIM_CSV = INTERIM_DIR / "pmc_validated.csv"
PROCESSED_CSV = PROCESSED_DIR / "pmc_monthly_model_input.csv"


def ensure_dirs() -> None:
    for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_and_validate() -> pd.Series:
    if not RAW_CSV.exists():
        try:
            RAW_CSV.write_bytes(SOURCE_CSV.read_bytes())
        except PermissionError:
            # Google Drive may deny writes in newly created synced subfolders.
            # In that case the root CSV remains the raw source of record.
            pass

    input_csv = RAW_CSV if RAW_CSV.exists() else SOURCE_CSV
    df = pd.read_csv(input_csv)
    required = {"date", "pmc_volume_index"}
    missing_cols = sorted(required.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate monthly dates found in source CSV.")

    monthly_index = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    missing_dates = monthly_index.difference(pd.DatetimeIndex(df["date"]))
    if len(missing_dates) > 0:
        raise ValueError(f"Missing monthly dates: {[d.strftime('%Y-%m-%d') for d in missing_dates]}")

    df["pmc_volume_index"] = pd.to_numeric(df["pmc_volume_index"], errors="coerce")
    if df["pmc_volume_index"].isna().any():
        raise ValueError("Missing/non-numeric values found in pmc_volume_index.")

    pmc_df = df[["date", "pmc_volume_index"]].copy()
    df.to_csv(INTERIM_CSV, index=False)
    pmc_df.to_csv(RAW_PMC_CSV, index=False)
    pmc_df.to_csv(PROCESSED_CSV, index=False)

    report = {
        "n_observations": int(len(df)),
        "start": df["date"].min().strftime("%Y-%m-%d"),
        "end": df["date"].max().strftime("%Y-%m-%d"),
        "frequency": "monthly_start",
        "missing_dates": [],
        "missing_values": df.isna().sum().to_dict(),
        "raw_columns": list(df.columns),
        "model_columns": ["date", "pmc_volume_index"],
    }
    (LOG_DIR / "validacao_dados_pmc.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    series = pmc_df.set_index("date")["pmc_volume_index"].asfreq("MS")
    return series


def save_descriptive_outputs(series: pd.Series) -> None:
    desc = series.describe(percentiles=[0.25, 0.5, 0.75]).to_frame("pmc_volume_index")
    extra = pd.DataFrame(
        {
            "pmc_volume_index": {
                "start": series.index.min().strftime("%Y-%m-%d"),
                "end": series.index.max().strftime("%Y-%m-%d"),
                "n_observations": len(series),
                "frequency": "monthly",
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
            }
        }
    )
    pd.concat([desc, extra]).to_csv(TABLE_DIR / "estatisticas_descritivas_pmc.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    series.plot(ax=ax, color="#1f5a85", linewidth=1.8)
    ax.set_title("Indice mensal de volume de vendas da PMC")
    ax.set_xlabel("Data")
    ax.set_ylabel("Indice de volume")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "serie_pmc.png", dpi=180)
    plt.close(fig)

    stl = STL(series, period=12, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle("Decomposicao STL da serie PMC", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decomposicao_stl_pmc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, lags=48, ax=axes[0])
    plot_pacf(series, lags=48, ax=axes[1], method="ywm")
    axes[0].set_title("ACF da serie em nivel")
    axes[1].set_title("PACF da serie em nivel")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acf_pacf_pmc.png", dpi=180)
    plt.close(fig)

    month_df = series.to_frame("pmc_volume_index")
    month_df["month"] = month_df.index.month
    fig, ax = plt.subplots(figsize=(10, 5))
    month_df.boxplot(column="pmc_volume_index", by="month", ax=ax, grid=False)
    fig.suptitle("")
    ax.set_title("Distribuicao da PMC por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Indice de volume")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplot_mensal_pmc.png", dpi=180)
    plt.close(fig)


def stationary_tests(series: pd.Series) -> None:
    rows = []
    transforms = {
        "nivel": series,
        "diff_1": series.diff().dropna(),
        "diff_sazonal_12": series.diff(12).dropna(),
        "diff_1_e_sazonal_12": series.diff().diff(12).dropna(),
    }
    for name, transformed in transforms.items():
        adf = adfuller(transformed, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_pvalue, _, _ = kpss(transformed, regression="c", nlags="auto")
        rows.append(
            {
                "transformacao": name,
                "adf_stat": adf[0],
                "adf_pvalue": adf[1],
                "kpss_stat": kpss_stat,
                "kpss_pvalue": kpss_pvalue,
                "n_obs": len(transformed),
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "testes_estacionariedade.csv", index=False)


def candidate_specs() -> list[tuple[str, tuple[int, int, int], tuple[int, int, int, int]]]:
    return [
        ("ARIMA(0,1,1)", (0, 1, 1), (0, 0, 0, 0)),
        ("ARIMA(1,1,0)", (1, 1, 0), (0, 0, 0, 0)),
        ("ARIMA(1,1,1)", (1, 1, 1), (0, 0, 0, 0)),
        ("ARIMA(2,1,1)", (2, 1, 1), (0, 0, 0, 0)),
        ("ARIMA(1,1,2)", (1, 1, 2), (0, 0, 0, 0)),
        ("SARIMA(0,1,1)(0,1,1,12)", (0, 1, 1), (0, 1, 1, 12)),
        ("SARIMA(1,1,1)(0,1,1,12)", (1, 1, 1), (0, 1, 1, 12)),
        ("SARIMA(1,1,0)(1,1,0,12)", (1, 1, 0), (1, 1, 0, 12)),
        ("SARIMA(0,1,1)(1,1,0,12)", (0, 1, 1), (1, 1, 0, 12)),
        ("SARIMA(1,1,1)(1,1,1,12)", (1, 1, 1), (1, 1, 1, 12)),
        ("SARIMA(0,1,2)(0,1,1,12)", (0, 1, 2), (0, 1, 1, 12)),
        ("SARIMA(2,1,1)(0,1,1,12)", (2, 1, 1), (0, 1, 1, 12)),
        ("SARIMA(2,1,2)(0,1,1,12)", (2, 1, 2), (0, 1, 1, 12)),
        ("SARIMA(3,1,1)(0,1,1,12)", (3, 1, 1), (0, 1, 1, 12)),
        ("SARIMA(1,1,3)(0,1,1,12)", (1, 1, 3), (0, 1, 1, 12)),
        ("SARIMA(2,1,2)(0,1,2,12)", (2, 1, 2), (0, 1, 2, 12)),
        ("SARIMA(3,1,3)(0,1,2,12)", (3, 1, 3), (0, 1, 2, 12)),
    ]


def fit_model(series: pd.Series, order, seasonal_order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=400)


def fit_candidates(series: pd.Series, test_size: int = 24):
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    split_report = pd.DataFrame(
        [
            {
                "train_start": train.index.min().strftime("%Y-%m-%d"),
                "train_end": train.index.max().strftime("%Y-%m-%d"),
                "test_start": test.index.min().strftime("%Y-%m-%d"),
                "test_end": test.index.max().strftime("%Y-%m-%d"),
                "train_n": len(train),
                "test_n": len(test),
            }
        ]
    )
    split_report.to_csv(TABLE_DIR / "particao_treino_teste.csv", index=False)

    candidates = candidate_specs()
    rows = []
    fitted_train = {}
    for name, order, seasonal_order in candidates:
        try:
            result = fit_model(train, order, seasonal_order)
            pred = result.get_forecast(steps=len(test)).predicted_mean
            pred.index = test.index
            error = test - pred
            resid = result.resid.dropna()
            lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
            rows.append(
                {
                    "modelo": name,
                    "aic_treino": result.aic,
                    "bic_treino": result.bic,
                    "mae_teste": np.mean(np.abs(error)),
                    "rmse_teste": np.sqrt(np.mean(np.square(error))),
                    "mape_teste_pct": np.mean(np.abs(error / test)) * 100,
                    "n_params": int(result.params.shape[0]),
                    "converged": bool(result.mle_retvals.get("converged", False)),
                    "ljung_box_pvalue_lag_12_treino": lb.loc[12, "lb_pvalue"],
                    "ljung_box_pvalue_lag_24_treino": lb.loc[24, "lb_pvalue"],
                }
            )
            fitted_train[name] = result
        except Exception as exc:
            rows.append(
                {
                    "modelo": name,
                    "aic_treino": np.nan,
                    "bic_treino": np.nan,
                    "mae_teste": np.nan,
                    "rmse_teste": np.nan,
                    "mape_teste_pct": np.nan,
                    "n_params": np.nan,
                    "converged": False,
                    "ljung_box_pvalue_lag_12_treino": np.nan,
                    "ljung_box_pvalue_lag_24_treino": np.nan,
                    "error": str(exc),
                }
            )

    results = pd.DataFrame(rows).sort_values(["rmse_teste", "mae_teste", "aic_treino"], na_position="last")
    results.to_csv(TABLE_DIR / "selecao_modelos_arima_sarima.csv", index=False)

    final_name = results.dropna(subset=["rmse_teste"]).iloc[0]["modelo"]
    order, seasonal_order = {
        name: (order, seasonal_order) for name, order, seasonal_order in candidates
    }[final_name]
    full_result = fit_model(series, order, seasonal_order)
    return final_name, full_result, fitted_train[final_name], train, test


def save_test_forecast(final_name: str, train_result, train: pd.Series, test: pd.Series) -> None:
    forecast = train_result.get_forecast(steps=len(test))
    pred = forecast.predicted_mean
    pred.index = test.index
    conf = forecast.conf_int(alpha=0.05)
    conf.index = test.index

    error = test - pred
    test_df = pd.DataFrame(
        {
            "date": test.index,
            "observed": test.values,
            "forecast": pred.values,
            "error": error.values,
            "absolute_error": np.abs(error.values),
            "absolute_percentage_error": np.abs(error.values / test.values) * 100,
            "lower_95": conf.iloc[:, 0].values,
            "upper_95": conf.iloc[:, 1].values,
        }
    )
    test_df.to_csv(TABLE_DIR / "previsao_teste_24m.csv", index=False)

    risk = pd.DataFrame(
        [
            {
                "modelo_final": final_name,
                "mae": test_df["absolute_error"].mean(),
                "rmse": np.sqrt(np.mean(np.square(test_df["error"]))),
                "mape_pct": test_df["absolute_percentage_error"].mean(),
                "max_abs_error": test_df["absolute_error"].max(),
                "test_start": test.index.min().strftime("%Y-%m-%d"),
                "test_end": test.index.max().strftime("%Y-%m-%d"),
                "test_n": len(test),
            }
        ]
    )
    risk.to_csv(TABLE_DIR / "risco_preditivo_teste_24m.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    train.loc["2017":].plot(ax=ax, label="Treino", color="#1f5a85", linewidth=1.8)
    test.plot(ax=ax, label="Observado no teste", color="#242424", linewidth=1.8)
    pred.plot(ax=ax, label="Previsao para o teste", color="#9b2f2f", linewidth=2)
    ax.fill_between(test.index, conf.iloc[:, 0], conf.iloc[:, 1], color="#9b2f2f", alpha=0.18)
    ax.axvline(test.index.min(), color="#555555", linestyle="--", linewidth=1)
    ax.set_title(f"Validacao preditiva da PMC - janela de teste 24m - {final_name}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Indice de volume")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "previsao_teste_pmc_24m.png", dpi=180)
    plt.close(fig)


def save_in_sample_selection_legacy(series: pd.Series) -> None:
    candidates = candidate_specs()

    rows = []
    fitted = {}
    for name, order, seasonal_order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fit_model(series, order, seasonal_order)
            resid = result.resid.dropna()
            lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
            rows.append(
                {
                    "modelo": name,
                    "aic": result.aic,
                    "bic": result.bic,
                    "n_params": int(result.params.shape[0]),
                    "converged": bool(result.mle_retvals.get("converged", False)),
                    "ljung_box_pvalue_lag_12": lb.loc[12, "lb_pvalue"],
                    "ljung_box_pvalue_lag_24": lb.loc[24, "lb_pvalue"],
                }
            )
            fitted[name] = result
        except Exception as exc:
            rows.append(
                {
                    "modelo": name,
                    "aic": np.nan,
                    "bic": np.nan,
                    "n_params": np.nan,
                    "converged": False,
                    "ljung_box_pvalue_lag_12": np.nan,
                    "ljung_box_pvalue_lag_24": np.nan,
                    "error": str(exc),
                }
            )

    results = pd.DataFrame(rows).sort_values(["aic", "bic"], na_position="last")
    results.to_csv(TABLE_DIR / "selecao_modelos_arima_sarima_amostra_completa.csv", index=False)


def save_diagnostics_and_forecast(series: pd.Series, final_name: str, result) -> None:
    resid = result.resid.dropna()
    lb = acorr_ljungbox(resid, lags=[6, 12, 18, 24], return_df=True)
    lb.to_csv(TABLE_DIR / "diagnostico_residuos.csv", index_label="lag")

    diag = pd.DataFrame(
        {
            "metrica": [
                "modelo_final",
                "media_residuos",
                "desvio_padrao_residuos",
                "assimetria_residuos",
                "curtose_residuos",
                "shapiro_pvalue",
            ],
            "valor": [
                final_name,
                resid.mean(),
                resid.std(),
                stats.skew(resid),
                stats.kurtosis(resid),
                stats.shapiro(resid.sample(min(len(resid), 500), random_state=123)).pvalue,
            ],
        }
    )
    diag.to_csv(TABLE_DIR / "resumo_modelo_final.csv", index=False)

    fig = result.plot_diagnostics(figsize=(12, 8))
    fig.suptitle(f"Diagnostico dos residuos - {final_name}", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnostico_residuos_modelo_final.png", dpi=180)
    plt.close(fig)

    forecast = result.get_forecast(steps=12)
    pred = forecast.predicted_mean
    conf = forecast.conf_int(alpha=0.05)
    fcst_df = pd.DataFrame(
        {
            "date": pred.index,
            "forecast": pred.values,
            "lower_95": conf.iloc[:, 0].values,
            "upper_95": conf.iloc[:, 1].values,
        }
    )
    fcst_df.to_csv(TABLE_DIR / "previsoes_12m.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    series.loc["2015":].plot(ax=ax, label="Observado", color="#1f5a85", linewidth=1.8)
    pred.plot(ax=ax, label="Previsao", color="#9b2f2f", linewidth=2)
    ax.fill_between(pred.index, conf.iloc[:, 0], conf.iloc[:, 1], color="#9b2f2f", alpha=0.18)
    ax.set_title(f"Previsao da PMC - 12 meses - {final_name}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Indice de volume")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "previsao_pmc_12m.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    series = load_and_validate()
    save_descriptive_outputs(series)
    stationary_tests(series)
    final_name, result, train_result, train, test = fit_candidates(series)
    save_test_forecast(final_name, train_result, train, test)
    save_in_sample_selection_legacy(series)
    save_diagnostics_and_forecast(series, final_name, result)
    print(f"Pipeline concluido. Modelo final por RMSE no teste: {final_name}")


if __name__ == "__main__":
    main()
