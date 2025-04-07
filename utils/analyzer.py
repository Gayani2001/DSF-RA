class DataFrameAnalyzer:
    @staticmethod
    def print_unique_counts(df, columns):
        for column in columns:
            print(f"Unique {column} count is {len(df[column].unique())}")

    @staticmethod
    def print_total_sales(df, month):
        total_sales = df[df['Date'].dt.month == month]['FinalValue'].sum()
        print(f"The total sales in month {month} is {total_sales}.")
        print(f"The total sales in month {month} is {(total_sales/ 1e6).round(2)} million.")

    @staticmethod
    def print_top_contributors(df, groupby_column, sort_column, n=20):
        top_contributors = df.groupby(groupby_column)[sort_column].sum().sort_values(ascending=False).head(n)
        print(top_contributors)

    @staticmethod
    def print_info(df):
        print(df.info())

    @staticmethod
    def print_fraction_of_sales_no_location(df, month, location_columns=['Latitude', 'Longitude'], distributor_column='DistributorCode'):
        
        """
        Calculates and prints the sales contribution of distributors with no location data.

        Parameters:
        df (pd.DataFrame): The DataFrame to calculate sales from.
        distributor_column (str): The name of the distributor column in the DataFrame.
        null_rowsUser (list): The list of distributor codes with no location data.

        Returns:
        None
        """
        null_rowsUser = df.loc[df[location_columns].isnull().all(axis=1), distributor_column].unique()
        print('Dealers with no location data ')
        print(null_rowsUser)
        total_sales = df[df['Date'].dt.month == month]['FinalValue'].sum()
        total_sales_no_location = df[df[distributor_column].isin(null_rowsUser)]['FinalValue'].sum()
        # print(f'They Contribute {total_sales_no_location} ({((total_sales_no_location*1e-6).round(2))} million) amount of sales')
        print(f'They Contribute {total_sales_no_location} ({((total_sales_no_location*1e-6).round(2))} million) amount of sales')
        fraction = ((total_sales_no_location / total_sales)*100).round(3)
        print(f"The fraction of sales from Dealers with no location data over total sales in month {month} is {fraction} %.")
