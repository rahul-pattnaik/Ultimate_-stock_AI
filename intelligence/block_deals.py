def detect_block_deal(volume, avg_volume):

    if volume > avg_volume * 3:
        return "Possible Block Deal"

    return "Normal"