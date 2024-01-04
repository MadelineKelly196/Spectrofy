def categorize_dance(value):
    if 0 <= value < 0.25:
        return "Low"
    elif 0.25 <= value < 0.5:
        return "Medium"
    elif 0.5 <= value < 0.75:
        return "High"
    else:
        return "Very High"