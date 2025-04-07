dfGeoCustomer = pd.read_csv('../all-geo-test-1/AllGeo (1).csv')

dfSFA_Orders = pd.read_excel('../task-1-dataset-test-1/Given Datasets/SFA_Orders_202.xlsx')

#GPS DATA JANUARY ONLY
dfSFA_GPSDataJan = pd.read_csv('../SFA_GPSData_202_2023January.csv')
dfSFA_GPSDataFeb = pd.read_csv('../SFA_GPSData_202_2023February.csv')
dfSFA_GPSDataMar1 = pd.read_csv('../SFA_GPSData_202_2023March_1.csv')
dfSFA_GPSDataMar2 = pd.read_csv('../SFA_GPSData_202_2023March_2.csv')
dfSFA_GPSDataMar3 = pd.read_csv('../SFA_GPSData_202_2023March_3.csv')
dfSFA_GPSDataMar4 = pd.read_csv('../SFA_GPSData_202_2023March_4.csv')

# Get all sheet names
SFA_Orders_xls = pd.ExcelFile('../SFA_Orders_202.xlsx')
sheet_names = SFA_Orders_xls.sheet_names

# Read each sheet into a separate DataFrame
dfSFA_Orders = {sheet: pd.read_excel(SFA_Orders_xls, sheet_name=sheet) for sheet in sheet_names}

dfSFA_OrdersJan = dfSFA_Orders['Jan']
dfSFA_OrdersFeb = dfSFA_Orders['Feb']
dfSFA_OrdersMar = dfSFA_Orders['Mar']
dfSFA_OrdersJan.head()
