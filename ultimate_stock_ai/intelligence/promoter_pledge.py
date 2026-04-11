def pledge_risk(promoter_pledge_percent):

    if promoter_pledge_percent > 50:
        return "High Risk"

    if promoter_pledge_percent > 20:
        return "Medium Risk"

    return "Low Risk"