import numpy as np

training_area = np.array([800, 900, 1000, 1100, 1200, 1300, 1400])
training_mean = training_area.mean()
training_std = training_area.std()

live_area = np.array([1800, 1900, 2000, 2100, 2200, 2300, 2400])
live_mean = live_area.mean()

threshold = 2 * training_std
difference = abs(live_mean - training_mean)

print("Training mean:", round(training_mean, 2))
print("Live mean:", round(live_mean, 2))
print("Mean difference:", round(difference, 2))
print("Drift threshold:", round(threshold, 2))

if difference > threshold:
    print("WARNING: Possible data drift detected")
else:
    print("Data distribution appears stable")
