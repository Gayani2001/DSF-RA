import pandas as pd
import logging

class DataPreprocessor:
    """
    A class used to preprocess data.

    ...

    Static Methods
    -------
    preprocess(df, columns) -> DataFrame
        Preprocesses the DataFrame by selecting columns and converting 'DistributorCode' to string.
    drop(df, columns) -> DataFrame
        Drops the specified columns from the DataFrame.
    copy_and_preprocess(df, columns) -> DataFrame
        Creates a copy of the DataFrame and preprocesses it.
    copy_and_drop(df, columns) -> DataFrame
        Creates a copy of the DataFrame and drops the specified columns.
    merge_dataframes(df1, df2, **kwargs) -> DataFrame
        Merges two DataFrames based on the specified keyword arguments.
    """

    @staticmethod
    def preprocess(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Preprocesses the DataFrame by selecting columns, converting 'DistributorCode' to string,
        and converting 'RecievedDate' to datetime format and extracting the date part.

        Parameters:
        df (pd.DataFrame): The DataFrame to preprocess.
        columns (list): The columns to select.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        try:
            df = df.loc[:, columns]
            df.loc[:, 'DistributorCode'] = df['DistributorCode'].astype(str)

            # Convert 'RecievedDate' to datetime format and extract the date part
            # df['Date'] = pd.to_datetime(df['Date'])
            # df['DateOnly'] = df['Date'].dt.date

            return df
        except Exception as e:
            logging.error("Error in preprocess method: %s", str(e))
            raise


    @staticmethod
    def drop(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to drop columns from.
        columns (list): The columns to drop.

        Returns:
        pd.DataFrame: The DataFrame with the columns dropped.
        """
        try:
            columns_to_drop = [col for col in columns if col in df.columns]
            return df.drop(columns=columns_to_drop)
        except Exception as e:
            logging.error("Error in drop method: %s", str(e))
            raise

    @staticmethod
    def copy_and_preprocess(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Creates a copy of the DataFrame and preprocesses it.

        Parameters:
        df (pd.DataFrame): The DataFrame to copy and preprocess.
        columns (list): The columns to select in the preprocessing step.

        Returns:
        pd.DataFrame: The copied and preprocessed DataFrame.
        """
        return DataPreprocessor.preprocess(df.copy(), columns)

    @staticmethod
    def copy_and_drop(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Creates a copy of the DataFrame and drops the specified columns.

        Parameters:
        df (pd.DataFrame): The DataFrame to copy and drop columns from.
        columns (list): The columns to drop.

        Returns:
        pd.DataFrame: The copied DataFrame with the columns dropped.
        """
        return DataPreprocessor.drop(df.copy(), columns)

    @staticmethod
    def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Merges two DataFrames based on the specified keyword arguments.

        Parameters:
        df1 (pd.DataFrame): The first DataFrame to merge.
        df2 (pd.DataFrame): The second DataFrame to merge.
        **kwargs: Keyword arguments to pass to the pandas merge function. Can optionally include a 'drop_column' argument to specify a column to drop from the merged DataFrame.

        Returns:
        pd.DataFrame: The merged DataFrame.
        """
        try:
            drop_column = kwargs.pop('drop_column', None)
            merged_df = df1.merge(df2, **kwargs)
            if drop_column:
                merged_df = merged_df.drop(columns=[drop_column])
            return merged_df
        except Exception as e:
            logging.error("Error in merge_dataframes method: %s", str(e))
            raise

    @staticmethod
    def dropna(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops the rows where any of the specified columns in the DataFrame have missing values.

        Parameters:
        df (pd.DataFrame): The DataFrame to drop rows from.
        columns (list): The columns to check for missing values.

        Returns:
        pd.DataFrame: The DataFrame with the rows dropped.
        """
        try:
            return df.dropna(subset=columns)
        except Exception as e:
            logging.error("Error in dropna method: %s", str(e))
            raise
