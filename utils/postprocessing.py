def categorize_dance(value):
    if 0 <= value < 0.4:
        return "Low"
    elif 0.4 <= value < 0.8:
        return "Moderate"
    else:
        return "High"