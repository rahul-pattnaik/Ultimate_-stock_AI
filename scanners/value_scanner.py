def undervalued(pe, industry_pe):

    if pe < industry_pe * 0.7:
        return True

    return False