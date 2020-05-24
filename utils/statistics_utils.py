import statistics


def quartiles(dataPoints):
    sortedPoints = sorted(dataPoints)
    mid = len(sortedPoints) // 2
    if len(sortedPoints) % 2 == 0:
        lowerQ = sortedPoints[:mid]
        higherQ = sortedPoints[mid:]
    else:
        lowerQ = sortedPoints[:mid]
        higherQ = sortedPoints[mid+1:]

    return median(lowerQ), median(sortedPoints), median(higherQ)

def median(dataPoints):
    sortedPoints = sorted(dataPoints)
    mid = len(sortedPoints) // 2
    if len(sortedPoints) % 2 == 0:
        med = sortedPoints[mid]
    else:
        med = (sortedPoints[mid] + sortedPoints[mid + 1]) / 2
    return med

def trimean(dataPoints):
    q1, q2, q3 = quartiles(dataPoints)
    return (q1 + 2*q2 + q3) / 4


def variance(dataPoints):
    return statistics.variance(dataPoints)

# def ttest(dataPoints, p_val):
