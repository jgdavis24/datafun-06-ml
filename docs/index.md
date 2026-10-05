# Penguin Mass Regression

> Which penguin measurement best predicts body mass? Testing three candidates
> on the same split instead of picking one and hoping.

## The Question

The example project predicts body mass from bill length. That choice is never
tested against the alternatives, and the dataset carries three numeric
measurements that could all plausibly work:

- bill length
- bill depth
- flipper length

This project fits all three on the same train/test split, compares them, and
then builds the full model around whichever one wins.

## Results

Flipper length wins by a wide margin.

| Feature | Test R² | Test RMSE |
|---|---|---|
| **flipper_length_mm** | **0.775** | **356.05** |
| bill_length_mm | 0.433 | 565.76 |
| bill_depth_mm | 0.208 | 668.28 |

![Test R-squared by candidate feature](./images/feature-comparison.png)

The example's choice came in second at roughly half the R².

Against a baseline that predicts the training mean for every test row
(RMSE 751.68), the flipper length model cuts average error by 53%. The bill
length model only cuts it by 25%.

The learned line:

```text
body_mass_g = 49.851 * flipper_length_mm - 5816.874
```

Each additional millimeter of flipper is worth about 50 grams of body mass.
The intercept of -5817 has no real meaning here, since a penguin with a
zero-length flipper does not exist.

![Flipper length vs body mass](./images/regression-predictions.png)

The scatter sits close to the line across the whole range, which is what a
0.775 looks like in practice.

## What The Metrics Did Not Show

The residual plots for the two features behave differently, and only one of
them tells you that.

![Residuals for the flipper length model](./images/regression-residuals.png)

With bill length, the residuals fanned out as bill length increased. They
stayed within about 700 grams around 37mm and spread from -1300 to +1000 by
50mm. That model was unreliable specifically for larger penguins.

With flipper length, the residuals sit in an even band of roughly plus or
minus 750 across the whole range, with no obvious widening.

Nothing in the bill length model's R² or RMSE flagged that pattern. It was
only visible in the plot.

## Limitation

Gentoo penguins are larger on every measurement in this dataset. Some of the
0.775 may be the model learning which species a penguin is rather than
learning a flipper-to-mass relationship.

In a prior project on this same data I found a pooled correlation of 0.871
drop to 0.468 for Adelie once the species were separated. Fitting one model
per species is the obvious next experiment.

## Method

```text
OBSERVE     344 penguins, 8 columns
DECLARE     target body_mass_g, three candidate features
PREPARE     drop rows missing feature or target (342 remain)
SPLIT       80/20, random_state=42
BASELINE    DummyRegressor, mean strategy
TRAIN       LinearRegression
PREDICT     on X_test (69 rows)
EVALUATE    baseline vs model, RMSE and R-squared
VISUALIZE   feature comparison, predictions, residuals
ASSESS      written interpretation in the log
```

All three candidate features use the same random seed, so they are judged on
identical training and test rows. Comparing them on different splits would
not be a fair test.

The baseline R² comes back at -0.002 rather than exactly zero. The baseline
predicts the training mean on test rows, and the two means are not identical,
so it scores slightly worse than predicting the test mean would.

## Data

Palmer Penguins, 344 rows, one row per penguin. Two rows drop out for missing
measurements, leaving 342 for modeling. See the
[Data Card](./data-card.md) for provenance and column descriptions.

## Professional Workflow

See [**Workflow B: Apply Example Project**](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
to get a project like this running on your machine.

## Documentation Index

- **Home** - this landing page
- [**Project Instructions**](./project-instructions.md)
- [**Concepts**](./concepts.md)
- [**Data Card**](./data-card.md)
- [**API**](./api.md)
