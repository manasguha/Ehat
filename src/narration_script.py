"""
Narration script for the Charging Station Load Prediction demo video.
7 sections, ~70-80 seconds total narration.
"""

NARRATION_SECTIONS = [
    {
        "id": "title",
        "text": (
            "Charging Station Load Prediction. "
            "Predicting electrical load from time of day and vehicle count."
        ),
    },
    {
        "id": "problem",
        "text": (
            "Electric vehicle charging stations face unpredictable demand. "
            "Operators need to forecast electrical load to manage grid capacity, "
            "prevent overloads, and optimize energy procurement. "
            "We model load as a function of two key variables: time of day and number of vehicles."
        ),
    },
    {
        "id": "data",
        "text": (
            "Our dataset contains two thousand one hundred sixty hourly observations "
            "simulated over ninety days. Load ranges from approximately two to three hundred kilowatts. "
            "Vehicle counts vary from zero to sixty. "
            "The data uses a Gaussian mixture arrival model to capture realistic peak and off-peak patterns."
        ),
    },
    {
        "id": "methodology",
        "text": (
            "We engineered cyclic time features using sine and cosine encoding "
            "to capture the circular nature of daily patterns. "
            "Features are standardized with a scaler before training. "
            "We evaluate three regression approaches: Linear Regression, "
            "Polynomial Regression, and Random Forest."
        ),
    },
    {
        "id": "results",
        "text": (
            "Random Forest achieves the best performance with an R squared of zero point nine five three two, "
            "a mean absolute error of seven point seven five kilowatts, "
            "and a root mean squared error of twelve point five three kilowatts. "
            "The polynomial model follows closely at R squared zero point nine five one three."
        ),
    },
    {
        "id": "load_curve",
        "text": (
            "The predicted load curve shows a clear daily pattern: "
            "low demand during early morning hours, rising through the morning, "
            "peaking in the evening around six to eight PM when vehicles return from commutes, "
            "then declining overnight."
        ),
    },
    {
        "id": "conclusion",
        "text": (
            "This project demonstrates that tree-based ensemble methods "
            "effectively capture the nonlinear relationship between time, vehicle count, and electrical load. "
            "The model supports real-time capacity planning for charging station operators. "
            "Built with Python, scikit-learn, and open source tools."
        ),
    },
]
