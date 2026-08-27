"""Explanation Formatter Module for AI Risk Manager.
Razorpay AI Buildathon — AI Risk Manager Track.

Converts technical model features, SHAP values, anomaly scores, and rule triggers
into transparent, merchant-understandable natural language explanations.
"""

from typing import Dict, Any, List, Tuple


FEATURE_DISPLAY_NAMES = {
    "amt": "Transaction Amount",
    "amt_to_user_avg_ratio": "Cardholder Spending Ratio",
    "amt_to_cat_median_ratio": "Category Spend Multiplier",
    "haversine_distance_km": "Terminal Geolocation Distance",
    "trans_velocity_1h": "1-Hour Transaction Velocity",
    "trans_velocity_24h": "24-Hour Transaction Velocity",
    "is_night_transaction": "Late Night / Off-Hours Timing",
    "hour": "Transaction Hour",
    "day_of_week": "Day of Week",
    "customer_age": "Cardholder Age",
    "category_fraud_rate": "Merchant Category Historical Risk",
}


def get_feature_display_name(feature_name: str) -> str:
    """Returns human-readable name for a given feature."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace("_", " ").title())


def format_feature_context(feature_name: str, feature_value: float, shap_value: float) -> str:
    """
    Generates a contextual natural-language explanation based on feature value
    and the direction/magnitude of its SHAP impact.
    """
    display_name = get_feature_display_name(feature_name)
    direction = "elevated fraud risk" if shap_value > 0 else "reduced risk (protective)"

    if feature_name == "amt":
        if shap_value > 0:
            return f"High transaction amount (${feature_value:.2f}) significantly {direction}."
        else:
            return f"Standard transaction amount (${feature_value:.2f}) {direction}."

    elif feature_name == "amt_to_user_avg_ratio":
        if shap_value > 0:
            return f"Order value is {feature_value:.1f}x higher than this customer's historical average spend ({direction})."
        else:
            return f"Spending aligns with customer's typical purchase history ({feature_value:.1f}x average, {direction})."

    elif feature_name == "amt_to_cat_median_ratio":
        if shap_value > 0:
            return f"Amount is {feature_value:.1f}x the median ticket size for this merchant sector ({direction})."
        else:
            return f"Ticket size is typical for this merchant category ({feature_value:.1f}x median, {direction})."

    elif feature_name == "haversine_distance_km":
        if shap_value > 0:
            return f"Terminal location is {feature_value:.1f} km away from cardholder's home address ({direction})."
        else:
            return f"Terminal is within normal local proximity ({feature_value:.1f} km from home, {direction})."

    elif feature_name == "trans_velocity_1h":
        if shap_value > 0:
            return f"Card velocity spike: {int(feature_value)} authorization(s) within the last hour ({direction})."
        else:
            return f"Normal velocity: no rapid authorization bursts detected ({direction})."

    elif feature_name == "trans_velocity_24h":
        if shap_value > 0:
            return f"Elevated 24-hour volume: {int(feature_value)} transactions in 24 hours ({direction})."
        else:
            return f"Routine daily transaction volume ({direction})."

    elif feature_name == "is_night_transaction":
        if int(feature_value) == 1 and shap_value > 0:
            return f"Transaction authorized during high-risk off-hours (midnight to 5 AM, {direction})."
        else:
            return f"Transaction initiated during standard daytime business hours ({direction})."

    elif feature_name == "category_fraud_rate":
        pct = feature_value * 100.0
        if shap_value > 0:
            return f"Merchant sector has elevated historical dispute rate ({pct:.2f}%, {direction})."
        else:
            return f"Merchant operates in a historically low-risk sector ({pct:.2f}% dispute rate, {direction})."

    else:
        return f"{display_name} = {feature_value:.2f} ({direction})."


def format_anomaly_reason(anomaly_score: float, anomaly_features: Dict[str, float]) -> str:
    """Generates an explainable summary of why the Isolation Forest scored a transaction."""
    if anomaly_score >= 70.0:
        drivers = []
        if anomaly_features.get("haversine_distance_km", 0) > 300:
            drivers.append(f"large geographical jump ({anomaly_features['haversine_distance_km']:.0f} km)")
        if anomaly_features.get("amt_to_user_avg_ratio", 1) > 3.0:
            drivers.append(f"atypical spending surge ({anomaly_features['amt_to_user_avg_ratio']:.1f}x user average)")
        if anomaly_features.get("amt", 0) > 500:
            drivers.append(f"abnormally large order value (${anomaly_features['amt']:.2f})")
        if anomaly_features.get("is_night_transaction", 0) == 1:
            drivers.append("late-night authorization")

        if drivers:
            return f"Unsupervised Isolation Forest detected severe behavioral outlier driven by: {', '.join(drivers)}."
        return "Unsupervised Isolation Forest detected a multi-dimensional statistical deviation from typical human spending baselines."
    elif anomaly_score >= 40.0:
        return "Transaction exhibits mild variance from standard behavioral boundaries, but remains within plausible consumer habits."
    else:
        return "Transaction patterns are fully consistent with typical, routine human spending behavior."

