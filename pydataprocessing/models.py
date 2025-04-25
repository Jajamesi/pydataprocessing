import warnings
from collections import defaultdict
import pandas as pd
from typing import List, Any

# set class for meta file
class metadata_obj:
    """
        Initialize the class attributes.
        Args:
            None
        Returns:
            None
    """
    def __init__(self):
        # default features
        self.var_names = list()
        self.original_vars = list()
        self.var_names_to_labels = defaultdict(lambda: None)
        self.var_labels_to_names = defaultdict(lambda: None)
        self.var_value_labels = defaultdict(lambda: None)
        self.var_measure = defaultdict(lambda: None)
        self.var_formats = defaultdict(lambda: None)
        self.var_types = defaultdict(lambda: None)
        self.var_display_widths = defaultdict(lambda: None)
        self.var_alignments = defaultdict(lambda: None)

        # type according to type_vars()
        self.q_type = defaultdict(lambda: None)

        # metadata for pure scale vars
        self.scale_vars = list()
        self.scale_recode_for_mean = defaultdict(lambda: None)
        self.scale_recode_for_aggregate = defaultdict(lambda: None)
        self.scale_value_labels = defaultdict(lambda: None)

        self.zero_vars = list()
        self.recode_pairs = defaultdict(lambda: None)

        # metadata for half scale vars
        self.half_scale_vars = list()
        self.half_scale_recode_for_aggregate = defaultdict(lambda: None)
        self.half_scale_value_labels = defaultdict(lambda: None)

        # metadata for awareness vars
        self.awareness_vars = list()

        # export features
        self.file_name = str()
        self.file_extension = str()


    def unprinted_symbol_clear(self):
        """
        This function replaces certain characters in var names and labels
        to remove unprinted symbols.
        """

        def is_not_number(value):
            """Check if the value is a number (int or float)."""
            try:
                float(value)  # Try converting to float
                return True
            except (ValueError, TypeError):
                return False

        # Define a dictionary with characters to be replaced
        replace_chars = {
            '\u00A0': ' ',
            '\u200B': '',
            '\\u200b': '',
            '\xa0': '',
            '\n': ' '
        }

        # Iterate over the var_names_to_labels dictionary
        # and replace the characters in the values and delete extra "?"
        self.var_names_to_labels = {
            key: ''.join(replace_chars.get(char, char) for char in value).strip("? ") +
                 ('?' if ((value.strip().endswith('?') and not value.strip().startswith('?'))
                      or (value.strip().endswith('??') and value.strip().startswith('?'))) else '')
                            if is_not_number(value) else str(value)
                                for key, value in self.var_names_to_labels.items()
        }

        # Iterate over the var_value_labels dictionary
        # and replace the characters in the values and delete extra "?"
        self.var_value_labels = {
            out_key: {
                key: ''.join(replace_chars.get(char, char) for char in value).strip().strip("? ")
                if is_not_number(value) else str(value)
                for key, value in self.var_value_labels[out_key].items()
            }
            for out_key, inner_dict in self.var_value_labels.items()
        }


    def fill_zero_labels(self):
        for q_name in self.var_value_labels:
            if self.var_value_labels.get(q_name).get(0) is None:
                self.var_value_labels.get(q_name).update({0: 'Не выбран'})
                self.zero_vars.append(q_name)

    def update(self, df, custom_meta=None):
        """
        This function update meta with properties of the new vars.
        Args: pandas dataframe.
        kwargs: custom meta - metadata obj.
        Returns: None.
        """

        # update var names as df columns
        self.var_names = list(df.columns)
        if not hasattr(self, 'var_alignments'):
            self.var_alignments = defaultdict(lambda: None)
        if not hasattr(self, 'var_display_widths'):
            self.var_display_widths = defaultdict(lambda: None)

        # if custom meta is provided, use it primarily
        if custom_meta:
            dict_attrs = [var for var in custom_meta.__dict__ if isinstance(custom_meta.__dict__[var], dict)]
            for attr in dict_attrs:
                self.__dict__[attr].update(custom_meta.__dict__[attr])
            print("custom meta updated")

            # self.var_names_to_labels.update(custom_meta.var_names_to_labels)
            # self.var_value_labels.update(custom_meta.var_value_labels)
            # self.var_measure.update(custom_meta.var_measure)
            # self.var_formats.update(custom_meta.var_formats)
            # self.var_types.update(custom_meta.var_types)



        for var in self.var_names:
            var_is_numeric = df[var].dtype in ['int64', 'float64']

            self.var_names_to_labels.setdefault(var, var)
            self.var_measure.setdefault(var, 'nominal')
            self.var_formats.setdefault(var, b'F8.2' if var_is_numeric else b'A12000')
            self.var_types.setdefault(var, 0 if var_is_numeric else 12000)
            self.var_alignments.setdefault(var, 'left' if self.var_types[var] > 0 else 'center')
            self.var_display_widths.setdefault(var, 10)

    def rename_vars(self, replacement_dict: dict) -> None:
        """
        This function replaces var names in metadata dictionaries.
        Args:
            replacement_dict (dict): A dictionary with the old var names as keys and the new var names as values.
        """

        def replace_dict_elements(data, replacement_dict):
            if isinstance(data, dict):
                return {replacement_dict.get(key, key): replace_dict_elements(value, replacement_dict) for key, value in
                        data.items()}
            elif isinstance(data, list):
                return [replace_dict_elements(item, replacement_dict) for item in data]
            else:
                raise TypeError("Unsupported data type: {}".format(type(data)))

        properties = [self.var_names, self.var_names_to_labels, self.var_value_labels,
                     self.var_measure, self.var_formats, self.var_types,
                     self.var_display_widths, self.var_alignments]

        for property in properties:
            property = replace_dict_elements(property, replacement_dict)

        # {replacement_dict.get(key, key): value for key, value in dictionary.items()}
        pass

    # method set dictionary with types and column names
    def type_vars(self, df: pd.DataFrame) -> None:
        """
        Assigns types to vars based on certain conditions.

        Args:
            df (pd.DataFrame): The data frame containing the vars.

        Returns:
            None
        """

        # List of technical vars
        technical_vars = [
            'CollectorNM',
            'respondent_id',
            'collector_id',
            'date_created',
            'date_modified',
            'survey_time',
            'ip_address',
        ]

        def get_element_by_index(index: int, lst = self.var_names) -> Any:
            """
            Get an element from a list by index.

            Args:
                index (int): The index of the element.
                lst (List): The list to get the element from.

            Returns:
                Any: The element at the specified index, or None if the index is out of range.
            """
            if 0 <= index < len(lst):
                return lst[index]
            else:
                return None

        def check_multipunch(name: str) -> bool:
            """
            Check if a var is a multipunch var.

            Args:
                name (str): The name of the var.

            Returns:
                bool: True if the var is a multipunch var, False otherwise.
            """
            if len(self.var_value_labels.get(name)) == 2:
                # Get the previous and next var names
                previous_var = get_element_by_index(self.var_names.index(name) - 1)
                next_var = get_element_by_index(self.var_names.index(name) + 1)

                # Check if the var labels match the previous or next var labels
                match_var_labels_list = [
                    self.var_names_to_labels.get(name) == var
                    for var in [self.var_names_to_labels.get(previous_var),
                                     self.var_names_to_labels.get(next_var)]
                ]

                # Check if the value 1 of the current var is different from the value 1 of the previous
                # or next var, and if the var labels match the previous or next var labels
                if any(match_var_labels_list):
                    value_1 = self.var_value_labels.get(name).get(1)
                    prev_value_1 = self.var_value_labels.get(previous_var, {}).get(1)
                    next_value_1 = self.var_value_labels.get(next_var, {}).get(1)

                    if (value_1 != prev_value_1 and match_var_labels_list[0]) or \
                            (value_1 != next_value_1 and match_var_labels_list[1]):
                        return True
            return False

        # Iterate over each var name
        for name in self.var_names:
            # Check if the var is a technical var
            if name in technical_vars:
                # Assign 'technical' type to the var
                self.q_type[name] = 'technical'
            elif 'wt' in name.lower():
                # Assign 'weight' type to the var
                self.q_type[name] = 'weight'
            elif not self.var_value_labels.get(name):
                # Assign 'open_ended_other' or 'open_ended' type to the var based on the presence of 'other' in the name
                self.q_type[name] = 'open_ended_other' if 'other' in name else 'open_ended'

                non_nan_values = df[name].dropna()

                # Count the number of numeric values in the var
                numeric_count = pd.to_numeric(non_nan_values, errors='coerce').notna().sum()

                total_count = len(non_nan_values)

                # Check if the var has more than 75% numeric values
                if total_count > 0 and not pd.isna(numeric_count) and numeric_count / total_count >= 0.75:
                    self.scale_vars.append(name)
            elif check_multipunch(name):
            # Assign 'multipunch' type to the var
                self.q_type[name] = 'multipunch'
            elif len(self.var_value_labels.get(name)) == 2 and \
                    self.var_names_to_labels.get(name).strip("? ").startswith(
                        self.var_value_labels.get(name).get(1).strip("? ")
                        )\
                    and name.startswith('q'):
                    # Assign 'info_screen' type to the var if var has 1 value, that is in var label and it is original
                self.q_type[name] = 'info_screen'
            else:
            # All others are singlepunches
                self.q_type[name] = 'singlepunch'


# Class of single bloc
class _block:
    def __init__(self):
        self.name = str()
        self.media = False
        self.questions = list()
        self.vars = list()
        self.map = list()

    def map_block(self):
        self.map = [q.type for q in self.questions]


# subclass of question for block
class _question:
    def __init__(self):
        self.number = int()
        self.type = str()
        self.q_label = str()
        self.sub_label = str()
        self.value_labels = list()
        self.variable = None