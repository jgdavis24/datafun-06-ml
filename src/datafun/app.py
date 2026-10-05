"""src/datafun/app.py - Project script.

Author: Josiah Davis
Date: 2026-10

HOW TO RUN THIS FILE:

From the VS Code menu (with only this project open in VS Code),
click "Terminal" / New Terminal to
open an integrated Terminal in the root project folder.
Paste the following command and press ENTER or RETURN
to run this file as a script:

uv run python -m datafun.app

DOMAIN:

A dataset of penguins.
See docs/data-card.md for more information about the dataset.

EXPLORE:

Earlier analysis showed relationships among
numeric penguin measurements.

In this project, we use one numeric feature
to predict one numeric target
with a simple linear regression model.

CUSTOM PROBLEM:

The example picks bill length as the feature without testing
whether it is the best available choice. Three numeric
measurements could plausibly predict body mass: bill length,
bill depth, and flipper length.

This project fits all three on the same train/test split and
compares them before committing to one, then builds the full
model around the winner.

A standard predictive modeling process is:

1. OBSERVE the data and prior findings.
2. DECLARE the target and feature.
3. PREPARE the modeling data.
4. SPLIT into training and test data.
5. BASELINE with a simple reference model.
6. TRAIN a LinearRegression model.
7. PREDICT on X_test.
8. EVALUATE baseline vs model on y_test.
9. VISUALIZE predictions and residuals.
10. ASSESS the results.

DESIGN:

Use this file to declare the data-specific choices
and the reasoning behind them,
then orchestrate the work.

Scikit-learn provides the machine learning tools.

The target, feature, split, baseline,
and model choices stay here because they are
analytical decisions specific to this project.
"""

# === DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
from ml_vizkit import save_chart
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# === CONFIGURE LOGGER ONCE FOR THE APPLICATION ===

LOG: logging.Logger = get_logger("P06", level="DEBUG")

# === DECLARE GLOBAL CONSTANTS ===

# Some global variables are CONSTANT.
# They do NOT change while the program runs.
# By convention, constants use UPPERCASE_WITH_UNDERSCORES.
# Final indicates that the value should not be reassigned.

# === LOCATE THE DATA FILE ===

DATA_FILE_PATH: Final[Path] = Path("data") / "raw" / "penguins.csv"

# === LOCATE THE CHART OUTPUT ===

CHART_DIR: Final[Path] = Path("docs") / "images"

PREDICTION_CHART_PATH: Final[Path] = CHART_DIR / "regression-predictions.png"

RESIDUAL_CHART_PATH: Final[Path] = CHART_DIR / "regression-residuals.png"

COMPARISON_CHART_PATH: Final[Path] = CHART_DIR / "feature-comparison.png"

# === DETERMINE WHAT ONE ROW REPRESENTS ===

GRAIN: Final[str] = "one penguin"

# === DECLARE THE TARGET ===

# CUSTOM: Choose one NUMERIC target value to predict.
# This must match a numeric column name EXACTLY
# as it appears in the data file.

TARGET_COLUMN: Final[str] = "body_mass_g"

# === CUSTOM: DECLARE THE CANDIDATE FEATURES ===

# Every numeric measurement that could reasonably
# predict body mass. All three get tested on the
# same split before one is chosen.

CANDIDATE_FEATURES: Final[list[str]] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
]

# === DECLARE THE FEATURE ===

# CUSTOM: Choose one NUMERIC feature
# that might help predict the target.
# This must match a numeric column name EXACTLY
# as it appears in the data file.

FEATURE_COLUMN: Final[str] = "flipper_length_mm"

# === DOCUMENT WHY THE FEATURE MIGHT HELP ===

# CUSTOM: Document the reasoning behind the feature choice.
# Do not assume the feature will work well.
# The model and evaluation will provide evidence.

FEATURE_DECISION: Final[str] = r"""
I want to predict body mass.

The example used bill length. I did not want to assume
that was the best choice, so I fit all three numeric
measurements on the same train/test split first and
compared their test R-squared.

Flipper length won, so the rest of this project builds
around it.

A flipper is a load-bearing limb and scales with overall
body size, while a bill is a feeding structure that can
vary with diet and species without tracking mass as closely.
That is a reason the result might make sense, not a reason
to have assumed it in advance.
"""

# === DECLARE THE TRAIN / TEST SPLIT ===

# CUSTOM: Decide how much data should be held back for testing.
# The test data should NOT be used to train the model.

TEST_FRACTION: Final[float] = 0.20

# CUSTOM: Choose whether the random split should be reproducible.
# A fixed random seed makes the same split each time the script runs.

RANDOM_SEED: Final[int] = 42

# === DOCUMENT THE SPLIT DECISION ===

# CUSTOM: Document the reasoning behind BOTH choices.
# The fraction and random seed should not be unexplained numbers.

SPLIT_DECISION: Final[str] = r"""
I will use 80% of the modeling rows for training
and hold back 20% for testing.

I want most of the available data to be available
for learning the model,
while still keeping a separate set of observations
that the model did not see during training.

The test rows will be used later
to evaluate how the trained model performs
on unseen observations.

I will use a random seed of 42.

The specific value 42 is not analytically important.
I use a fixed seed so the random split is reproducible.
Running the project again will produce the same
training and test observations,
which makes results easier to reproduce and compare.

The same seed is used for the feature comparison, so all
three candidates are judged on identical training and test
rows. Comparing them on different splits would not be a
fair test.
"""

# === DECLARE THE BASELINE ===

BASELINE_STRATEGY: Final[str] = "mean"

# === DOCUMENT THE BASELINE DECISION ===

BASELINE_DECISION: Final[str] = r"""
Before evaluating the LinearRegression model,
I need a simple baseline for comparison.

The baseline will ignore flipper length
and predict the average body mass
from the training data for every test observation.

A useful predictive model should improve
on this simple reference prediction.
"""

# === DOCUMENT THE MODEL DECISION ===

MODEL_DECISION: Final[str] = r"""
I will use LinearRegression.

Linear regression fits a straight-line relationship
between the selected feature and target.

This gives a simple and interpretable model
that can be compared with the baseline.

Fitting a line does not prove that a straight line
is a good description of the relationship.

The evaluation metrics and residual plot
will help assess whether the model is useful.
"""


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Entry point when running this file as a Python script.

    This is where the instructions begin.

    Arguments: None.
    Returns: None.
    """
    log_header(LOG, "P06 - LINEAR REGRESSION")

    LOG.info("===================================")
    LOG.info("START main()")
    LOG.info("===================================")

    LOG.info("-------------------------------")
    LOG.info("01. OBSERVE the data and prior findings.")
    LOG.info("-------------------------------")

    log_path(LOG, "data file", path=DATA_FILE_PATH)

    df: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)

    LOG.info("Data loaded successfully.")
    LOG.info(f"Grain: {GRAIN}")
    LOG.info(f"Rows: {df.shape[0]}")
    LOG.info(f"Columns: {df.shape[1]}")
    LOG.info(f"Column names: {df.columns.tolist()}")

    LOG.info("-------------------------------")
    LOG.info("02. DECLARE the target and feature.")
    LOG.info("-------------------------------")

    # === CUSTOM: COMPARE CANDIDATE FEATURES FIRST ===

    # Fit one model per candidate feature on the same split
    # so the comparison is fair, then pick the winner.

    LOG.info("CUSTOM: comparing candidate features on the same split.")

    comparison_results: dict[str, dict[str, float]] = {}

    for candidate in CANDIDATE_FEATURES:
        candidate_df: pd.DataFrame = df.dropna(subset=[candidate, TARGET_COLUMN]).copy()

        candidate_X: pd.DataFrame = candidate_df[[candidate]]
        candidate_y: pd.Series = candidate_df[TARGET_COLUMN]

        (
            candidate_X_train,
            candidate_X_test,
            candidate_y_train,
            candidate_y_test,
        ) = train_test_split(
            candidate_X,
            candidate_y,
            test_size=TEST_FRACTION,
            random_state=RANDOM_SEED,
        )

        candidate_model = LinearRegression()
        candidate_model.fit(candidate_X_train, candidate_y_train)
        candidate_predictions = candidate_model.predict(candidate_X_test)

        comparison_results[candidate] = {
            "rmse": float(
                root_mean_squared_error(candidate_y_test, candidate_predictions)
            ),
            "r_squared": float(r2_score(candidate_y_test, candidate_predictions)),
        }

    for candidate, scores in comparison_results.items():
        LOG.info(
            f"  {candidate}: "
            f"R-squared {scores['r_squared']:.3f}, "
            f"RMSE {scores['rmse']:.2f}"
        )

    LOG.info(f"Target:  {TARGET_COLUMN}")
    LOG.info(f"Feature: {FEATURE_COLUMN}")
    LOG.info(FEATURE_DECISION)

    LOG.info("-------------------------------")
    LOG.info("03. PREPARE the modeling data.")
    LOG.info("-------------------------------")

    # A regression model requires a value
    # for both the selected feature and target.
    # Keep the original DataFrame unchanged.
    # Create a separate modeling DataFrame
    # containing complete feature / target pairs.

    required_columns: list[str] = [
        FEATURE_COLUMN,
        TARGET_COLUMN,
    ]

    df_model: pd.DataFrame = df.dropna(subset=required_columns).copy()

    count_original: int = df.shape[0]
    count_model: int = df_model.shape[0]
    count_dropped: int = count_original - count_model

    LOG.info(f"Original rows: {count_original}")
    LOG.info(f"Modeling rows: {count_model}")
    LOG.info(f"Rows dropped: {count_dropped}")

    # scikit-learn expects X to be a 2-dimensional
    # feature matrix and y to be a 1-dimensional target.

    X: pd.DataFrame = df_model[[FEATURE_COLUMN]]
    y: pd.Series = df_model[TARGET_COLUMN]

    LOG.info(f"X shape: {X.shape}")
    LOG.info(f"y shape: {y.shape}")

    LOG.info("-------------------------------")
    LOG.info("04. SPLIT into training and test data.")
    LOG.info("-------------------------------")

    LOG.info(SPLIT_DECISION)

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
    )

    LOG.info(f"Training rows: {X_train.shape[0]}")
    LOG.info(f"Test rows: {X_test.shape[0]}")

    LOG.info("-------------------------------")
    LOG.info("05. BASELINE with a simple reference model.")
    LOG.info("-------------------------------")

    LOG.info(BASELINE_DECISION)

    baseline_model = DummyRegressor(
        strategy=BASELINE_STRATEGY,
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    baseline_predictions: np.ndarray = baseline_model.predict(X_test)

    baseline_rmse: float = float(
        root_mean_squared_error(
            y_test,
            baseline_predictions,
        )
    )

    baseline_r_squared: float = float(
        r2_score(
            y_test,
            baseline_predictions,
        )
    )

    LOG.info(f"Baseline strategy: {BASELINE_STRATEGY}")
    LOG.info(f"Baseline RMSE: {baseline_rmse:.2f}")
    LOG.info(f"Baseline R-squared: {baseline_r_squared:.3f}")

    LOG.info("-------------------------------")
    LOG.info("06. TRAIN a LinearRegression model.")
    LOG.info("-------------------------------")

    LOG.info(MODEL_DECISION)

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    slope: float = float(model.coef_[0])
    intercept: float = float(model.intercept_)

    LOG.info("The model learned this line:")
    LOG.info(f"{TARGET_COLUMN} = {slope:.3f} * {FEATURE_COLUMN} + {intercept:.3f}")

    LOG.info("-------------------------------")
    LOG.info("07. PREDICT on X_test.")
    LOG.info("-------------------------------")

    # The model has never trained on X_test.
    # Use the trained model to predict target values
    # for these held-back observations.

    model_predictions: np.ndarray = model.predict(X_test)

    LOG.info(f"Predictions created: {len(model_predictions)}")

    LOG.info("-------------------------------")
    LOG.info("08. EVALUATE baseline vs model on y_test.")
    LOG.info("-------------------------------")

    # RMSE measures prediction error
    # in the same units as the target.
    # Lower RMSE is better.

    model_rmse: float = float(
        root_mean_squared_error(
            y_test,
            model_predictions,
        )
    )

    # R-squared describes how much of the variation
    # in the test target is accounted for by the model.
    # Larger values generally indicate a better fit.

    model_r_squared: float = float(
        r2_score(
            y_test,
            model_predictions,
        )
    )

    LOG.info("BASELINE RESULTS")
    LOG.info(f"RMSE:      {baseline_rmse:.2f}")
    LOG.info(f"R-squared: {baseline_r_squared:.3f}")

    LOG.info("LINEAR REGRESSION RESULTS")
    LOG.info(f"RMSE:      {model_rmse:.2f}")
    LOG.info(f"R-squared: {model_r_squared:.3f}")

    LOG.info("-------------------------------")
    LOG.info("09. VISUALIZE predictions and residuals.")
    LOG.info("-------------------------------")

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # === CUSTOM: FEATURE COMPARISON CHART ===

    _comparison_figure, comparison_ax = plt.subplots()

    comparison_labels: list[str] = list(comparison_results.keys())
    comparison_scores: list[float] = [
        comparison_results[name]["r_squared"] for name in comparison_labels
    ]

    comparison_ax.bar(comparison_labels, comparison_scores)

    comparison_ax.set_title("Test R-squared by Candidate Feature")
    comparison_ax.set_xlabel("Feature")
    comparison_ax.set_ylabel("R-squared on Test Data")
    plt.setp(comparison_ax.get_xticklabels(), rotation=15, ha="right")

    save_chart(
        comparison_ax,
        COMPARISON_CHART_PATH,
    )

    LOG.info(f"Chart saved successfully at {COMPARISON_CHART_PATH}.")

    # === PREDICTIONS CHART ===

    # Plot the actual test observations.

    _prediction_figure, prediction_ax = plt.subplots()

    x_test_values: np.ndarray = X_test[FEATURE_COLUMN].to_numpy()
    y_test_values: np.ndarray = y_test.to_numpy()

    prediction_ax.scatter(
        x_test_values,
        y_test_values,
        label="Actual",
    )

    # Sort x values so the regression line
    # is drawn from left to right.

    prediction_order: np.ndarray = np.argsort(x_test_values)

    prediction_ax.plot(
        x_test_values[prediction_order],
        model_predictions[prediction_order],
        label="Predicted",
    )

    # CUSTOM: The analyst can customize
    # the returned Matplotlib Axes object.

    prediction_ax.set_title("Flipper Length vs. Body Mass")
    prediction_ax.set_xlabel("Flipper Length (mm)")
    prediction_ax.set_ylabel("Body Mass (g)")
    prediction_ax.legend()

    save_chart(
        prediction_ax,
        PREDICTION_CHART_PATH,
    )

    LOG.info(f"Chart saved successfully at {PREDICTION_CHART_PATH}.")

    # === RESIDUAL CHART ===

    # A residual is:
    #
    # actual value - predicted value
    #
    # Residuals near zero indicate predictions
    # close to the observed target values.

    residuals: np.ndarray = y_test_values - model_predictions

    _residual_figure, residual_ax = plt.subplots()

    residual_ax.scatter(
        x_test_values,
        residuals,
    )

    # Draw a horizontal reference line at zero.

    residual_ax.axhline(0)

    # CUSTOM: The analyst can customize
    # the returned Matplotlib Axes object.

    residual_ax.set_title("Residuals for Flipper Length Model")
    residual_ax.set_xlabel("Flipper Length (mm)")
    residual_ax.set_ylabel("Residual (Actual - Predicted Body Mass)")

    save_chart(
        residual_ax,
        RESIDUAL_CHART_PATH,
    )

    LOG.info(f"Chart saved successfully at {RESIDUAL_CHART_PATH}.")

    # ============================================================
    # 10. ASSESS
    # ============================================================

    LOG.info("-------------------------------")
    LOG.info("10. ASSESS the results.")
    LOG.info("-------------------------------")

    # Run this app first.
    # Review the baseline and model metrics.
    # Review both visualizations.
    # Then record your CUSTOM observations
    # in a simple multi-line raw string.

    LOG.info(r"""CUSTOM OBSERVATIONS:
    I tested three candidate features on the same split before
    choosing one.

        flipper_length_mm   R-squared 0.775   RMSE 356.05
        bill_length_mm      R-squared 0.433   RMSE 565.76
        bill_depth_mm       R-squared 0.208   RMSE 668.28

    Flipper length won by a wide margin. The example's choice of
    bill length came in second at roughly half the R-squared.

    Against the baseline RMSE of 751.68, the flipper length model
    came in at 356.05, cutting average error by 53%. The bill
    length model only cut it by 25%.

    The learned line is:

        body_mass_g = 49.851 * flipper_length_mm - 5816.874

    Each additional millimeter of flipper is worth about 50 grams
    of body mass. The intercept of -5817 has no real meaning here,
    since a penguin with a zero-length flipper does not exist.

    The residual plots differ in a way the metrics do not show.
    When I ran this with bill length, the residuals fanned out as
    bill length increased, staying within about 700 grams at 37mm
    and spreading from -1300 to +1000 by 50mm. With flipper length
    they sit in an even band of roughly plus or minus 750 across
    the whole range.

    That matters because the bill length model was unreliable
    specifically for larger penguins, and nothing in its R-squared
    or RMSE said so. I only saw it in the plot.

    Based on this evidence, flipper length is the right single
    predictor for body mass in this dataset.

    Next, I would like to fit the model separately for each species.
    Gentoo penguins are larger on every measurement, so some of the
    0.775 may be the model learning which species a penguin is
    rather than learning a flipper-to-mass relationship. In a prior
    project I found a pooled correlation of 0.871 dropped to 0.468
    for Adelie once the species were separated.
    """)

    # ============================================================
    # DISPLAY
    # ============================================================

    LOG.info("In a script, call plt.show() at the end to display all charts.")
    LOG.info("Close all chart windows (with the close button) to continue.")

    plt.show()

    LOG.info("===================================")
    LOG.info("END main() - Executed successfully!")
    LOG.info("===================================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
