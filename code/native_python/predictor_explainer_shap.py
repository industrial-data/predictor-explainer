"""Run Predictor Explainer SHAP analysis from JMP 19.

Expected globals supplied by JSL:
    pe_input_dt
    pe_x_cols
    pe_y_cols
    pe_time_cols
    pe_weight_cols
    pe_add_time_features  (1 = derive calendar features from the time column;
                           JSL sets it only when the user gives the timestamp
                           both as X and as t, Datetime — otherwise time-based
                           predictors would leak the target through trends)
    pe_n_trees
    pe_signal_to_noise
    pe_add_diffs
    pe_diff_rows

This script does not install packages and does not use intermediate data files.
It publishes result data frames back into JMP with jmp.from_dataframe().
"""

import re
import time
import traceback
import warnings

import jmp
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

try:
    from lightgbm.sklearn import LGBMRegressor
except Exception:
    # On Mac the LightGBM wheel needs the OpenMP runtime (brew install libomp);
    # without it the slower scikit-learn RandomForest is used instead
    LGBMRegressor = None
    print("LightGBM is not available; using the scikit-learn RandomForest fallback.")

warnings.filterwarnings("ignore", category=FutureWarning)
# pandas 3 deprecates the dataframe interchange protocol that jmp.from_dataframe consumes
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if str(value):
        return [str(value)]
    return []


def _as_scalar(value, default=None):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    if value is None:
        return default
    return value


def _get_global(name, default=None):
    return globals().get(name, default)


def _to_pandas(data_table):
    if isinstance(data_table, pd.DataFrame):
        return data_table.copy()
    return pd.api.interchange.from_dataframe(data_table).copy()


def _safe_column_map(columns):
    used = set()
    mapping = {}
    reverse = {}

    for original in columns:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(original)).strip("_") or "column"
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        used.add(candidate)
        mapping[original] = candidate
        reverse[candidate] = str(original)

    return mapping, reverse


def _normalise_for_display(frame):
    q03 = frame.quantile(0.03)
    q97 = frame.quantile(0.97)
    denom = (q97 - q03).replace(0, np.nan)
    return ((frame - q03) / denom).clip(0, 1).fillna(0)


def _coerce_jmp_datetime(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return pd.to_datetime(numeric, unit="s", origin=pd.Timestamp("1904-01-01"), errors="coerce")

    return pd.to_datetime(series, errors="coerce")


def _add_time_features(x_frame, time_series):
    parsed = _coerce_jmp_datetime(time_series)
    if parsed.dropna().empty:
        print("Time column could not be parsed; skipping time features.")
        return x_frame

    if parsed.dropna().dt.year.head(5).mean() < 1960:
        print("Time column looks like an index rather than a date; skipping time features.")
        return x_frame

    x_frame = x_frame.copy()
    x_frame["day_of_month"] = parsed.dt.day
    x_frame["day_of_week"] = parsed.dt.dayofweek
    x_frame["day_of_year"] = parsed.dt.dayofyear
    x_frame["week_of_year"] = parsed.dt.isocalendar().week.astype(float)
    x_frame["week_of_month"] = np.floor(parsed.dt.day / 7)
    x_frame["month"] = parsed.dt.month
    x_frame["quarter"] = parsed.dt.quarter
    x_frame["semester"] = np.ceil(parsed.dt.month / 6)
    x_frame["year"] = parsed.dt.year
    x_frame["hour"] = parsed.dt.hour
    x_frame["minute"] = parsed.dt.minute
    return x_frame


def _fit_tree_model(x_frame, y_series, weights, n_trees):
    if LGBMRegressor is not None:
        model = LGBMRegressor(n_estimators=n_trees, verbosity=-1)
        model_x = x_frame
        model_name = "LightGBM"
    else:
        model = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
        model_x = x_frame.fillna(x_frame.median(numeric_only=True)).fillna(0)
        model_name = "RandomForestRegressor"

    if weights is not None:
        model.fit(model_x, y_series, sample_weight=np.asarray(weights).flatten())
    else:
        model.fit(model_x, y_series)

    return model, model_x, model_name


def _publish_table(frame, name):
    jmp_frame = frame.reset_index(drop=True).copy()
    for col in jmp_frame.columns:
        if pd.api.types.is_object_dtype(jmp_frame[col]) or pd.api.types.is_string_dtype(jmp_frame[col]):
            jmp_frame[col] = jmp_frame[col].astype("string").fillna("")

    # jmp.from_dataframe takes only the dataframe; JSL hides the tables after Python Get
    table = jmp.from_dataframe(jmp_frame)
    try:
        table.name = name
    except Exception:
        print(f"Could not rename result table to {name}; keeping the default name.")
    return table


def _run_analysis():
    # JSL retrieves the result tables from the Python globals with Python Get()
    global pe_global_shap_dt, pe_shap_values_dt, pe_shap_plot_dt

    tic = time.time()

    input_df = _to_pandas(_get_global("pe_input_dt"))
    x_cols = _as_list(_get_global("pe_x_cols"))
    y_cols = _as_list(_get_global("pe_y_cols"))
    time_cols = _as_list(_get_global("pe_time_cols"))
    weight_cols = _as_list(_get_global("pe_weight_cols"))

    if not x_cols:
        raise ValueError("Select at least one X column.")
    if not y_cols:
        raise ValueError("A Y column (target) is required for the SHAP analysis.")

    missing_cols = [col for col in x_cols + y_cols + time_cols + weight_cols if col not in input_df.columns]
    if missing_cols:
        raise ValueError("Columns were not found in the transferred JMP table: " + ", ".join(missing_cols))

    add_time_features = int(_as_scalar(_get_global("pe_add_time_features"), 0)) == 1
    n_trees = max(1, int(_as_scalar(_get_global("pe_n_trees"), 100)))
    signal_to_noise = float(_as_scalar(_get_global("pe_signal_to_noise"), 1.0))
    add_diffs = int(_as_scalar(_get_global("pe_add_diffs"), 1)) == 1
    diff_rows = max(1, int(_as_scalar(_get_global("pe_diff_rows"), 1)))

    x_original = input_df[x_cols].apply(pd.to_numeric, errors="coerce")
    mapping, reverse_mapping = _safe_column_map(x_original.columns)
    x_model = x_original.rename(columns=mapping)
    display_name = {safe: reverse_mapping.get(safe, safe) for safe in x_model.columns}

    if "row_index" in input_df.columns:
        row_index = pd.to_numeric(input_df["row_index"], errors="coerce").fillna(0).astype(int)
        row_index = row_index.rename("SHAP_row_index")
    else:
        row_index = pd.Series(np.arange(1, len(input_df) + 1), name="SHAP_row_index")

    y_name = y_cols[0]
    y = pd.to_numeric(input_df[y_name], errors="coerce").rename(y_name)
    if y_name in mapping:
        x_model = x_model.drop(columns=[mapping[y_name]])

    time_series = None
    if time_cols:
        # the timestamp itself is kept out of the model and only carried
        # through for the SHAP plot tables
        time_series = input_df[time_cols[0]].rename(time_cols[0])
        if add_time_features:
            x_model = _add_time_features(x_model, time_series)
            for col in x_model.columns:
                display_name.setdefault(col, col)
        else:
            print(
                "Time features skipped: give the timestamp as X and as t, Datetime to add them."
            )

    if add_diffs:
        x_diff = x_model.diff(periods=diff_rows, axis=0)
        x_diff.columns = [f"Diff_{col}" for col in x_diff.columns]
        for col in x_model.columns:
            display_name[f"Diff_{col}"] = f"Diff_{display_name.get(col, col)}"
        x_model = pd.concat([x_model, x_diff], axis=1)

    x_model = x_model.select_dtypes(include=np.number).dropna(axis=1, how="all")
    x_model = x_model.replace([np.inf, -np.inf], np.nan)
    if x_model.shape[1] == 0:
        raise ValueError("No numeric X columns remain after cleaning.")

    weights = None
    if weight_cols:
        raw_weights = pd.to_numeric(input_df[weight_cols[0]], errors="coerce")
        if raw_weights.notna().any():
            if raw_weights.max() != raw_weights.min():
                weights = (raw_weights - raw_weights.min()) / (raw_weights.max() - raw_weights.min())
            else:
                weights = raw_weights.fillna(0)

    valid_index = y.dropna().index
    y = y.loc[valid_index]
    x_model = x_model.loc[valid_index, :]
    row_index = row_index.loc[valid_index].reset_index(drop=True)
    x_model = x_model.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if len(y) < 2:
        raise ValueError("At least two rows with nonmissing Y values are required.")
    if time_series is not None:
        time_series = time_series.loc[valid_index].reset_index(drop=True)
    if weights is not None:
        weights = weights.loc[valid_index].fillna(0).reset_index(drop=True)

    normal_noise = np.random.normal(size=len(x_model))
    uniform_noise = np.random.uniform(size=len(x_model))
    shuffled_y = y.to_numpy().copy()
    np.random.shuffle(shuffled_y)
    x_model["Normal_Noise"] = normal_noise
    x_model["Uniform_Noise"] = uniform_noise
    x_model["Shuffle_Yield_Noise"] = shuffled_y
    display_name["Normal_Noise"] = "Normal Noise"
    display_name["Uniform_Noise"] = "Uniform Noise"
    display_name["Shuffle_Yield_Noise"] = "Shuffle Yield Noise"

    model, model_x, model_name = _fit_tree_model(x_model, y, weights, n_trees)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(model_x)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    df_shap_values = pd.DataFrame(shap_values, columns=model_x.columns)
    shap_abs = np.abs(shap_values)
    mean_shap = shap_abs.mean(axis=0)
    std_shap = shap_abs.std(axis=0)
    mean_shap = np.where(mean_shap == 0, 1e-9, mean_shap)
    coef_variance = std_shap / mean_shap

    df_global_shap_values = pd.DataFrame(
        {
            "Tags": [display_name.get(col, col) for col in model_x.columns],
            "model_column": model_x.columns,
            "mean": mean_shap,
            "std": std_shap,
            "CoefVariance": coef_variance,
        }
    ).sort_values(by="mean", ascending=False)

    ordered_model_cols = df_global_shap_values["model_column"].tolist()
    noise_cols = ["Shuffle_Yield_Noise", "Normal_Noise", "Uniform_Noise"]
    noise_positions = [ordered_model_cols.index(col) for col in noise_cols if col in ordered_model_cols]
    noise_cut_position = min(noise_positions) if noise_positions else len(ordered_model_cols)
    noise_col = ordered_model_cols[noise_cut_position] if noise_positions else ordered_model_cols[-1]
    noise_impact = df_global_shap_values.set_index("model_column").loc[noise_col, "mean"]
    noise_cut_impact = signal_to_noise * noise_impact

    selected_model_cols = df_global_shap_values.loc[
        df_global_shap_values["mean"] > noise_cut_impact,
        "model_column",
    ].tolist()
    if noise_col not in selected_model_cols:
        selected_model_cols.append(noise_col)
    selected_model_cols = [col for col in selected_model_cols if col in model_x.columns]
    if not selected_model_cols:
        selected_model_cols = ordered_model_cols[: min(10, len(ordered_model_cols))]

    x_selected = model_x[selected_model_cols].copy()
    x_normalized = _normalise_for_display(x_selected)
    df_shap_values = df_shap_values[selected_model_cols].copy()
    df_shap_values.columns = [display_name.get(col, col) for col in selected_model_cols]
    x_selected.columns = [display_name.get(col, col) for col in selected_model_cols]
    x_normalized.columns = x_selected.columns

    df_global_shap_values = df_global_shap_values.iloc[0 : noise_cut_position + 1, :].drop(columns=["model_column"])

    df_shap_values.insert(0, y_name, y.values)
    if time_series is not None:
        df_shap_values.insert(0, time_series.name, time_series.values)
    df_shap_values.insert(0, "SHAP_row_index", row_index.values)

    id_vars = ["SHAP_row_index"]
    if time_series is not None:
        id_vars.append(time_series.name)
    id_vars.append(y_name)

    shap_melted = df_shap_values.melt(id_vars=id_vars, var_name="tag", value_name="shap(value)")
    x_melted = x_selected.melt(var_name="tag", value_name="value")
    x_norm_melted = x_normalized.melt(var_name="tag", value_name="norm(value)")
    df_shap_plot_table = pd.concat(
        [
            shap_melted.reset_index(drop=True),
            x_melted["value"].reset_index(drop=True),
            x_norm_melted["norm(value)"].reset_index(drop=True),
        ],
        axis=1,
    )

    pe_global_shap_dt = _publish_table(df_global_shap_values, "PE Global SHAP Values")
    pe_shap_values_dt = _publish_table(df_shap_values, "PE SHAP Values")
    pe_shap_plot_dt = _publish_table(df_shap_plot_table, "PE SHAP Plot Table")

    elapsed = round(time.time() - tic, 2)
    return (
        f"Created JMP result tables with {model_name}: "
        f"{pe_global_shap_dt.name}, {pe_shap_values_dt.name}, "
        f"{pe_shap_plot_dt.name}. "
        f"Rows: {len(y)}. Selected features: {len(selected_model_cols)}. Seconds: {elapsed}."
    )


# JSL reads this to tell the user when the slower fallback model was used
pe_model_fallback = 1 if LGBMRegressor is None else 0

try:
    pe_run_message = _run_analysis()
    pe_run_ok = 1
except Exception:
    pe_run_ok = 0
    pe_run_message = "Predictor Explainer Python run failed. Review the JMP log."
    print(traceback.format_exc())

print(pe_run_message)
