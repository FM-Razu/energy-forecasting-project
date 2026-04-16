# Solar Power Generation ML Project

## Exploratory Data Analysis

**Dataset**: ~10,500 rows × 2 columns (time + power), compiled from multiple 5-min interval solar panel readings across Utah/Colorado (2006).

| Feature | Type | Description |
|---------|------|-------------|
| `LocalTime` | datetime | Timestamp (M/D/YY H:MM) |
| `Power(MW)` | float | Real power output (0–172 MW) |

**Key Findings**:
- **Zero-inflated target**: Power(MW) is heavily skewed toward 0 (nighttime/low generation), with a long right tail up to 172 MW — regression models may need specialized handling for zero-inflated distributions.
- **No missing values**: Dataset is complete; no imputation needed.
- **Temporal structure**: Time-based features (hour, month) will likely be the strongest predictors of solar generation — diurnal cycle dominates.
- **Strong potential features**: Engineering hour-of-day and month from `LocalTime` should yield high correlation with target.

**Modeling implications**:
- Log or square-root transform of `Power(MW)` may help normalize the skewed distribution.
- Consider two-stage models (classification: is generation > 0? → regression: how much?).