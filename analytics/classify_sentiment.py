from openpyxl import load_workbook
from pandas.core.frame import DataFrame

filename = "data/Stay Positive - sample_1000_0_and_500_0_and_100_01_and_100_02_stage_two_features.xlsx"
print("Loading '%s'" % filename)
wb = load_workbook(filename)
ws = wb.active
df = DataFrame(ws.values)
print(len(df))
