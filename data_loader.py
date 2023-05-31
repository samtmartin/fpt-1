import csv
import random
import torch

from config import ColumnConfig
from config import DataConfig
from torch import Tensor

class DataLoader():

    def __init__(
            self, 
            data_config: DataConfig, 
            block_size: int) -> None:
        """
        Loads dataset for model training and evaluation. Supports batching, 
        partitioning, scaling, shuffling, etc.

        Args:
          -data_config: config used to determine the data inputs to provide to 
           the model and outputs that should be expected from the model's 
           `forward` method.
          -block_size: sequence length. Companies with fewer than block_size + 1 
           rows will be excluded from dataset.
        """

        # Dictionary (dict[str, list[tuple[int, int, str]]]) containing the
        # split partitions. Values are lists of tuples where first and second 
        # elements represent given Company's start and end index in `_data` and 
        # the third element represents the Company's ticker (stored for 
        # debugging and analysis).
        self._partitions = {}
        # Dictionary (dict[str, Tensor]) containing the sampling probabilities
        # for each split in `_partitions`. The sampling probs represent the 
        # expected probability of sampling the corresponding index in 
        # `_partitions` (e.g. if there are two companies in dataset, A and B,
        # and Company A has twice as much data, A will be sampled approximately 
        # twice as frequently). This ensures that each row of dataset has equal 
        # probability of being sampled.
        self._partition_probs = {}
        # Container for all of the data in dataset (i.e. input features). May 
        # include both scaled and non-scaled data.
        self._data = None
        # List (list[ColumnConfig]) of the expected model's output configs. 
        # Required to transform scaled data back to raw values.
        self._output_column_configs = None
        # List (list[int]) of the expected model output column indices within 
        # the `_data` tensor. Used by `get_batch` to get targets.
        self._output_column_indices = None
        # Sequence length.
        self._block_size = block_size

        # Builds the dataset.
        self.__set_instance_variables(data_config)

    def get_batch(
            self,
            split: str, 
            batch_size: int) -> tuple[Tensor]:
        probs = self._partition_probs[split]
        samples = torch.multinomial(probs, batch_size, replacement=True)

        x = []
        y = []
        for s in samples:
            start, end, ticker = self._partitions[split][s]
            i = random.randint(start, end - self._block_size)
            x.append(self._data[i:i + self._block_size])
            output_data = (self._data[:, self._output_column_indices])
            y.append(output_data[i+1:i + self._block_size + 1])

        return torch.stack(x), torch.stack(y)

    def get_partition(
            self, 
            split: str) -> Tensor:
        return self._partitions[split]

    def get_data(
            self) -> Tensor:
        return self._data
    
    def generate_raw_data(
            self, 
            scaled_data: Tensor) -> Tensor:
        for i, config in enumerate(self._output_column_configs):
            raw_data = scaled_data
            column = scaled_data[:, i]
            strategy = config.scaling_strategy

            if strategy == "mean":
                column = column * config.mean
            elif strategy == "standardization":
                column = (column * config.std) + config.mean
            elif strategy == "log":
                column = torch.exp(column) - torch.abs(config.scaling_adj)
            elif strategy == "log10":
                column = 10**column - torch.abs(config.scaling_adj)

            raw_data[:, i] = column

        return raw_data

    def get_num_input_features(
            self) -> int:
        return len(self._data[0])
    
    def get_num_output_features(
            self) -> int:
        return len(self._output_column_indices)
    
    def get_forecast_data(
            self, 
            num_predictions: int) -> tuple[Tensor, Tensor]:
        probs = self._partition_probs["test"]
        sample = torch.multinomial(probs, 1, replacement=True)[0]

        x = []
        y = []
        start, end, ticker = self._partitions["test"][sample]
        i = random.randint(start, end - (num_predictions + self._block_size))
        x.append(self._data[i:i + self._block_size])
        output_data = (self._data[:, self._output_column_indices])
        y_start = self._block_size + 1
        y_end = num_predictions + self._block_size + 1
        y.append(output_data[i+y_start:i + y_end])

        return torch.stack(x), torch.stack(y)
    
    def __set_instance_variables(
            self, 
            data_config: DataConfig) -> None:
        input_scaling_configs = self.__generate_input_scaling_configs(
            data_config)
        
        # Add financial data. Do not not change order of below operations 
        # without changing `__get_input_and_output_configs`.
        scaling_data, row_metadata = self.__generate_financial_data(data_config)

        # Maybe add macro data.
        if len(data_config.macro_columns) > 0:
            scaling_data = self.__add_macro_data(
                data_config, scaling_data, row_metadata)
            
        # Scale data. Must be done before adding other model inputs that should 
        # not be scaled.
        scaled_data, self._output_column_configs = self.__scale_data(
            scaling_data, input_scaling_configs)
        self._output_column_indices = [
            config.data_index for config in self._output_column_configs]
        
        non_scaling_data = self.__generate_non_scaling_data(
            data_config, row_metadata)
        
        if non_scaling_data.numel() > 0:
            self._data = torch.cat((scaled_data, non_scaling_data), dim=1)
        else:
            self._data = scaled_data

        self._partitions, self._partition_probs = self.__generate_partitions(
            data_config, row_metadata)
    
    def __generate_input_scaling_configs(
            self, 
            data_config: DataConfig) -> tuple[
        list[DataConfig], list[DataConfig]]:
        # Only the financial and macro data values are eligible for scaling.
        input_scaling_configs = []

        for config in data_config.financial_columns:
            input_scaling_configs.append(config)
            i = len(input_scaling_configs) - 1
            for _ in range(config.lag_steps):
                input_scaling_configs.append(config)
            if config.include_yoy_prediction:
                input_scaling_configs.append(config)

        for config in data_config.macro_columns:
            for _ in range(config.lag_steps):
                input_scaling_configs.append(config)

        return input_scaling_configs
    
    def __generate_financial_data(
            self, 
            data_config: DataConfig) -> tuple[
        list[list[float]], list[tuple[str, str, str]]]:
        # Read in the financial data.
        raw_financial_data = None
        with open('data/public_company_financials.csv') as csvfile:
            # Save contents of file to list in order to make data available 
            # outside the with block.
            raw_financial_data = [record for record in csv.reader(csvfile)]

        file_column_headers = raw_financial_data[0]
        # File column headers to indices.
        cti = {col: i for i, col in enumerate(file_column_headers)}

        # `financial_data` holds all the company-specific data for columns 
        # requested in `data_config`. `row_metadata` (company, data quarter 
        # ended, business sector) is metadata for every row in `financial_data`.
        financial_data = []
        row_metadata = []
        for i in range(1, len(raw_financial_data)): # Assumes header present.
            record = raw_financial_data[i]
            ticker = record[cti["ticker"]]
            quarter_ended = record[cti["quarter_ended"]]
            sector = record[cti["sector"]]
            row_metadata.append([ticker, quarter_ended, sector])

            row = []
            for config in data_config.financial_columns:
                value = record[cti[config.header]]
                # Missing data is expected along the below dimension. Consider 
                # experimenting with other forms of imputation (e.g. use 
                # Company's mean or median) or remove the entire record.
                if value == "" and config.header == "tev_ltm_revenue":
                    value = 0
                # Remove commas so `value` can be converted to float (e.g. 
                # "2,150.00"). 
                else:
                    value = value.replace(',', '')
                row.append(float(value))

                c = len(row) - 1 # Column index of value in `financial data`.
                # Maybe add lag for metric.
                for step in range(config.lag_steps):
                    step += 1 # Step is zero-indexed.
                    row_index = i - step - 1
                    if row_index < 0 or ticker != row_metadata[row_index][0]:
                        row.append(0.)
                    else:
                        row.append(financial_data[row_index][c])

                # Maybe add yoy predictions for metric.
                if config.include_yoy_prediction:
                    # i is one-indexed whereas `financial_data` is zero-indexed.
                    row_index = i - 1 
                    last_row_index = row_index - 4

                    if (last_row_index < 0 or 
                        ticker != row_metadata[last_row_index][0]):
                        row.append(0.)
                        continue
                    
                    # Current quarter.
                    cq = row[c]
                    # Current quarter's prior year.
                    cqpy = financial_data[row_index-4][c]
                    # Current quarter's year-over-year growth rate.
                    gr = cq / cqpy if cqpy != 0 else 1
                    # Next quarter's prior year.
                    nqpy = financial_data[row_index-3][c] 
                    prediction = nqpy * gr
                    row.append(prediction)

            financial_data.append(row)
        
        return financial_data, row_metadata
    
    def __get_key_and_value(
            self, 
            data: list[str], 
            i: int, 
            months_per_key: int) -> tuple[str, list[float]]:
        key = ""
        value = []
        # If `months_per_key` = 3, `p` represents the index of the first month 
        # in quarter.
        p = i - (months_per_key - 1)
        while p <= i:
            raw_key, string_value = data[p]

            value.append(float(string_value))
            if p == i:
                key = self.__get_key(raw_key)
            p += 1

        return key, value
    
    def __get_key(
            self, 
            date_string: str) -> str:
        """
        Arg: a string representing a date. Expects format "yyyy-mm-dd".
        Returns: the "yyyy-mm" portion of `date_string`.
        """

        return date_string[:7]
    
    def __build_data_dictionary(
            self, 
            path: str, 
            months_per_key: int) -> dict[str, list[float]]:
        """
        Reads in the data at `path`. Expects raw data to be monthly and to have 
        two columns (month and value) with month in format "yyyy-mm-dd". 
        Produces dictionary where each key ("yyyy-mm") is associated with a list 
        containing that month's value as well as the prior two month's values. 
        Three months are provided because a financial quarter is three months.
        """
        
        with open(path) as csvfile:
            data = [record for record in csv.reader(csvfile)]
        output = {}
        # Skip header and (`months_per_key` - 1) months of data given lookback
        # (e.g. if `months_per_key` = 3, the lookback period is 2).
        for i in range(months_per_key, len(data)):
            key, value = self.__get_key_and_value(data, i, months_per_key)
            output[key] = value
        return output
    
    def __add_macro_datum(
            self,
            mutable_scaling_data: list[list[float]],
            row_metadata: list[tuple[str, str, str]], 
            path: str, 
            lag_steps: int) -> None:
        macro_dict = self.__build_data_dictionary(path, lag_steps)
        for i, row in enumerate(row_metadata):
            key = self.__get_key(row[1]) # Quarter ended is at pos 1.
            mutable_scaling_data[i].extend(macro_dict[key])

    def __add_macro_data(
            self, 
            data_config: DataConfig,
            scaling_data: list[list[float]],
            row_metadata: list[tuple[str, str, str]]) -> list[list[float]]:
        mutable_scaling_data = scaling_data.copy()

        for config in data_config.macro_columns:
            if config.header == "three_month_tbill" and config.lag_steps > 0:
                self.__add_macro_datum(
                    mutable_scaling_data, row_metadata, 
                    'data/3_month_us_treasury_securities.csv', config.lag_steps)
            elif config.header == "ten_year_tbill" and config.lag_steps > 0:
                self.__add_macro_datum(
                    mutable_scaling_data, row_metadata, 
                    'data/10_year_us_treasury_securities.csv', config.lag_steps)
            elif (config.header == "personal_consumption" and 
                  config.lag_steps > 0):
                self.__add_macro_datum(
                    mutable_scaling_data, row_metadata, 
                    'data/personal_consumption_expenditures.csv', 
                    config.lag_steps)
            elif config.header == "trucking" and config.lag_steps > 0:
                self.__add_macro_datum(
                    mutable_scaling_data, row_metadata, 
                    'data/general_freight_trucking.csv', 
                    config.lag_steps)
                
        return mutable_scaling_data
    
    def __scale_column(
            self,
            column: Tensor, 
            column_config: ColumnConfig,
            is_output_column: bool) -> tuple[Tensor, ColumnConfig]:
        strategy = column_config.scaling_strategy
        if strategy == "mean":
            mean = column.mean()
            column_config.mean = mean
            column = column / mean
        elif strategy == "standardization":
            mean = column.mean()
            std = column.std()
            column_config.mean = mean
            column_config.std = std
            column = (column - mean) / std
        elif strategy == "log" or strategy == "log10":            
            min = torch.min(column)
            if min.item() < 1:
                min -= 1
                column = column + abs(min)
            else:
                min = torch.tensor(0.)
            
            if is_output_column == True:
                column_config.scaling_adj = min

            if strategy == "log":
                column = torch.log(column)
            else:
                column = torch.log10(column)
        return column, column_config
    
    def __scale_data(
            self,
            scaling_data: list[list[float]], 
            mutable_input_scaling_configs: list[ColumnConfig]) -> tuple[
        list[list[float]], list[DataConfig]]:
        data = torch.tensor(scaling_data)
        output_column_configs = []
        prev_config = None
        for i, config in enumerate(mutable_input_scaling_configs):
            is_output_column = (prev_config == None or 
                                    prev_config.header != config.header)
            scaled_column, config = self.__scale_column(
                data[:, i], config, is_output_column)
            data[:, i] = scaled_column
            # Only output the current quarter's metric (i.e. not lag steps or 
            # yoy predictions).
            if config.is_output and is_output_column:
                config.data_index = i
                output_column_configs.append(config)
            prev_config = config
        return data, output_column_configs
    
    def __generate_non_scaling_data(
            self, 
            data_config: DataConfig,
            row_metadata: list[tuple[str, str, str]]) -> Tensor:
        # Pre-build `sectors` so that we can one hot encode per-company sector
        # info when building `non_scaling_data`.
        sectors = set([])
        if data_config.include_sectors:
            for meta in row_metadata:
                sectors.add(meta[2])

        # Sort allows for stable one hot encodings.
        sectors = sorted(list(sectors))
        # Sector to index.
        sti = {sector: i for i, sector in enumerate(sectors)}
        
        non_scaling_data = []
        for meta in row_metadata:
            row = []
            # Maybe add quarter ended data.
            if data_config.include_quarter_ended:
                # Appends result of dividing the the month that the quarter 
                # ended by twelve. 
                row.append(int(meta[1].split("-")[1]) / 12)

            # Maybe add sector data.
            if data_config.include_sectors:
                one_hot_encoding = [0] * len(sectors)
                one_hot_encoding[sti[meta[2]]] = 1
                row.extend(one_hot_encoding)

            if len(row) > 0:
                non_scaling_data.append(row)

        return torch.tensor(non_scaling_data)
    
    def __is_end_of_batch(
            self, 
            current_ticker: str, 
            i: int, 
            row_metadata: list[tuple[str, str, str]]) -> bool:
        next_i = i + 1
        if next_i == len(row_metadata):
            return True

        next_row = row_metadata[next_i]
        next_ticker = next_row[0]
        if current_ticker != next_ticker:
            return True

        return False
    
    def __get_partition_probs(
            self, 
            partitions: list[tuple[int, int, str]]) -> Tensor:
        """
        Converts `partitions` into a Tensor containing the percent of total that 
        each partition represents. For example, if `partitions` is 
        [[0,24,"x"],[25,49,"y"],[50,99,"z"]], the output would be 
        [0.25,0.25,0.5].
        """

        lengths = torch.tensor(
            [end - start + 1 for start, end, ticker in partitions])
        sum = lengths.sum()
        return lengths.float() / sum
    
    def __generate_partitions(
            self, 
            data_config: DataConfig, 
            row_metadata: list[tuple[str, str, str]]) -> tuple[
        dict[str, list[tuple[int, int, str]]], dict[str, Tensor]]:
        # Partitions (start and end index) for each company. Required because 
        # the number of quarters represented per company varies, which makes a 
        # (B,T,C) shaped tensor that contains a batch for each co not possible.
        company_partitions = []
        num_elements = 0
        for i, row in enumerate(row_metadata):
            ticker = row[0]
            num_elements += 1
            if self.__is_end_of_batch(ticker, i, row_metadata):
              # Only add Company's data to `data` if the number of quarters 
              # exceeds `block_size`, which allows for sequence to be shifted to 
              # the right by 1 when forecasting.
              if num_elements > self._block_size:
                  start = i - (num_elements - 1)
                  end = i

                  # "part" is a special token representing data that has already 
                  # been partitioned (due to miss quarters).
                  if (data_config.split_company_data == False or 
                      ticker.find("part") > -1 or
                      int((end - start) // 2) <= self._block_size):
                    company_partitions.append([start, end, ticker])
                  else:
                      first_end = start + int((end - start) // 2)
                      second_start = first_end + 1
                      company_partitions.append([start, first_end, ticker])
                      company_partitions.append([second_start, end, ticker])
              
              num_elements = 0
        
        # Shuffle in order to make sector mix more balanced between splits.
        if data_config.shuffle_data:
            random.seed(1337)
            random.shuffle(company_partitions)

        # Create partitions dict.
        partitions = {}        
        tr = int(data_config.percent_train*len(company_partitions))
        percent_val_and_test = 1 - data_config.percent_train
        v = int(percent_val_and_test/2*len(company_partitions))
        partitions["train"] = company_partitions[:tr]
        partitions["val"] = company_partitions[tr:tr+v]
        partitions["test"] = company_partitions[tr+v:]
        
        # Create partition_probs dict.
        partition_probs = {}
        partition_probs["train"] = self.__get_partition_probs(
            partitions["train"])
        partition_probs["val"] = self.__get_partition_probs(partitions["val"])
        partition_probs["test"] = self.__get_partition_probs(
            partitions["test"])

        return partitions, partition_probs