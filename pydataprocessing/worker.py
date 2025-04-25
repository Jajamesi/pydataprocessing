import pandas as pd
pd.options.mode.copy_on_write = True

import numpy as np
from collections import defaultdict
from .models import metadata_obj
from typing import List, Any, Union
import warnings
import os
import tempfile
import shutil
import random
import string
import re
from thefuzz import fuzz
import difflib

import collections
import collections.abc

# Monkey-patch to fix the Iterable import issue in savReaderWriter
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
from savReaderWriter import SavReader, SavHeaderReader, SavWriter

def unique_no_nan(_x):
    """
    Returns a sorted list of unique values from the input Series `x`, excluding any NaN values.

    Parameters:
    _x (pandas.Series): The input Series.

    Returns:
    list: A sorted list of unique values from `x`, excluding any NaN values.
    """
    return sorted(list(_x.dropna().unique()))


def read_spss(file_path, unprinted_symbol_clear: bool = True):

    """
    Reads an SPSS file and returns a DataFrame with the data and _metadata.
    Parameters:
        file_path : The path to the SPSS file or bytes object.
    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing the DataFrame with the data and a dictionary with the _metadata.
    """

    # Define a helper function to decode bytes values
    def _decode(_val: Union[bytes, Any]) -> Union[str, Any]:
        """
        Decodes a _val from Windows-1251 encoding if it is of type bytes.

        Parameters:
            _val (Union[bytes, Any]): The _val to be decoded.

        Returns:
            Union[str, Any]: The decoded _val if it was of type bytes, otherwise the original _val.
        """
        if isinstance(_val, bytes):
            _val_decoded = _val.decode("Windows-1251", errors="replace").strip()
            return _val_decoded if _val_decoded != "" else np.nan
        return _val

    """
    This section works this way - we create a temporary file, write the .sav file to it, 
    read it with savReaderWriter and delete the temporary file. 
    This is crucial since savReaderWriter can't work around non ASCII characters in the path to file.
    """

    # streamlit work around - if file path is already bytes, just use it
    if isinstance(file_path, bytes):
        file_contents = file_path
        name, extension = "default", "sav"
    else:
        name, extension = os.path.splitext(os.path.basename(file_path))
        with open(file_path, 'rb') as file:
            file_contents = file.read()

    # Create a temporary file and write the contents to it
    with tempfile.NamedTemporaryFile(delete=False, mode='wb') as _temp_spss:
        _temp_spss_name = _temp_spss.name
        _temp_spss.write(file_contents)

    try:
        # Read the header of the SPSS file
        with SavHeaderReader(savFileName=_temp_spss_name, ioUtf8=False, ioLocale='russian') as _header:
            _metadata = _header.all()

        # Create a _metadata object
        _meta = metadata_obj()

        # Set the file name and extension
        _meta.file_name, _meta.file_extension = name, extension

        # Decode var names and filter out empty values
        _meta.var_names = [_decode(value) for value in _metadata.varNames if _decode(value)]
        _meta.original_vars = _meta.var_names

        # Decode var labels and fill empty values with var name
        _meta.var_names_to_labels = {
            _decode(key): _decode(value) if (_decode(value) and not pd.isna(_decode(value))) else _decode(key)
            for key, value in _metadata.varLabels.items()
        }

        # Decode value labels for vars and filter out empty values
        _meta.var_value_labels = {
            _decode(outer_key): {
                _decode(inner_key): _decode(inner_value)
                for inner_key, inner_value in _metadata.valueLabels[outer_key].items() if _decode(inner_value)
            }
            for outer_key in _metadata.valueLabels.keys()
        }

        # Decode var measure levels and filter out empty values
        _meta.var_measure = {
            _decode(key): _decode(value)
            for key, value in _metadata.measureLevels.items() if _decode(key)
        }

        # Filter out empty var formats
        _meta.var_formats = {
            _decode(key): value
            for key, value in _metadata.formats.items() if _decode(key)
        }

        # Filter out empty var types
        _meta.var_types = {
            _decode(key): value
            for key, value in _metadata.varTypes.items() if _decode(key)
        }

        # Set default display widths to 10 for all vars
        _meta.var_display_widths = defaultdict(lambda: None, {name: 10 for name in _meta.var_names})

        # Set default alignments to 'left' if var type is present, 'center' otherwise
        _meta.var_alignments = defaultdict(lambda: None, {
            name: 'left' if _meta.var_types.get(name) else 'center'
            for name in _meta.var_names
        })

        # clean unprinted symbols
        if unprinted_symbol_clear:
            _meta.unprinted_symbol_clear()


        # Read the data from the SPSS file
        with SavReader(
                savFileName=_temp_spss_name,
                returnHeader=False,
                verbose=False,
                selectVars=None,
                ioUtf8=False,
                ioLocale='russian',
                idVar=None
        ) as _reader:
            _data = _reader.all()
            _df = pd.DataFrame(_data) #.replace({None: np.nan})

        # Rename the columns of the DataFrame
        _df.columns = _meta.var_names

        # Decode string vars
        var_to_decode = [var for var in _meta.var_names if _meta.var_types.get(var)]
        _df.loc[:, var_to_decode] = _df.loc[:, var_to_decode].apply(lambda x:
                                                    x.apply(lambda value:
                                                            value.decode('Windows-1251', errors="ignore")
                                                            if value != b"" else np.nan))

        os.remove(_temp_spss_name)
        return _df, _meta
    except:
        os.remove(_temp_spss_name)

def write_spss(
        df,
        meta,
        path_to_export,
        export_file_name,
        extension="zsav", 
        custom_meta=None
):
    """
    Writes a DataFrame to an SPSS file.
    Args:
        _df (pandas.DataFrame): The DataFrame to be exported.
        _path_to_export (str): The path to the directory where the file will be exported.
        _export_file_name (str): The name of the file to be exported.
        _meta (object): The _meta object containing var information.
        extension (str, optional): The file extension. Defaults to "zsav".
        custom_meta (object, optional): The custom _meta object. Defaults to None.
    """

    # Avoid permutations of original dataframe
    _df=df.copy()
    _meta = meta

    # Update _meta
    _meta.update(_df, custom_meta)

    # Encode string values to Windows-1251
    def _encode(_value):
        return _value if not isinstance(_value, str) else _value.encode(
                                                            encoding = "Windows-1251",
                                                            errors= "ignore"
        )

    # Encode vars and their attributes
    _var_names = [_encode(var_name) for var_name in _meta.var_names]
    _var_types = {_encode(var_name): var_type for var_name, var_type in _meta.var_types.items()}

    _var_labels = {_encode(var_name): _encode(var_label) for var_name, var_label in _meta.var_names_to_labels.items()}
    _value_labels = {
        _encode(var_name): {
            val_code: _encode(val_label) for val_code, val_label in _meta.var_value_labels[var_name].items()
        }
        for var_name in _meta.var_value_labels.keys()
    }
    _formats = {_encode(var_name): _encode(var_format) for var_name, var_format in _meta.var_formats.items()}
    _measure_levels = {_encode(var_name): _encode(var_measure) for var_name, var_measure in _meta.var_measure.items()}
    _column_widths = {_encode(var_name): int(var_width) for var_name, var_width in _meta.var_display_widths.items()}
    _alignments = {_encode(var_name): _encode(var_align) for var_name, var_align in _meta.var_alignments.items()}

    # Encode dataframe var values to Windows-1251
    _vars_to_encode = [var for var in _meta.var_names if _meta.var_types.get(var)]
    _df[_vars_to_encode] = _df[_vars_to_encode].apply(lambda x:
                                                x.apply(lambda value:
                                                        str(value).encode(
                                                            encoding = "Windows-1251",
                                                            errors= "replace"
                                                        )
                                                        if not pd.isna(value) else b"")).astype(bytes)

    # Add extension to the file name
    _extension = f".{extension}"

    """
    This section works this way - we create a temporary file, write the DataFrame to it, 
    move it to the export path with user specified name and delete the temporary file. 
    This is crucial since savReaderWriter can't work around non ASCII characters in the path to file.
    """

    # get the temp directory
    _temp_dir = tempfile.gettempdir()
    # create name for the temporary file
    _characters = string.ascii_letters + string.digits
    _temp_file_name = ''.join(random.choice(_characters) for _ in range(5))

    # Create the export path
    _temp_export_path = os.path.join(_temp_dir, (_temp_file_name + _extension))

    # Write the DataFrame to the SPSS file
    with SavWriter(
            savFileName=_temp_export_path,
            varNames=_var_names,
            varTypes=_var_types,
            varLabels=_var_labels,
            valueLabels=_value_labels,
            formats=_formats,
            measureLevels=_measure_levels,
            columnWidths=_column_widths,
            alignments=_alignments,
            ioUtf8=False,
            ioLocale='russian'
    ) as _writer:
         _writer.writerows(_df)

    # Move the temporary file to the export path
    _new_file_path = os.path.join(path_to_export, (export_file_name + _extension))
    shutil.move(_temp_export_path, _new_file_path)

def excel_to_sav(_df,
                 path_to_export,
                 export_file_name,
                 extension="zsav",
                 external_order_path=None,
                 col_threshold=2,
                 tech_names = ['respondent_id', 'date_created'],
                 value_threshold=12,
                 mp_separator = " / ",
                 ):
    """
    Converts an Excel file to an SPSS file format (.zsav).

    Parameters:
    - _df: pandas DataFrame - The input DataFrame containing the data to be converted.
    - path_to_export: str - The path to the directory where the SPSS file will be exported.
    - export_file_name: str - The name of the exported SPSS file.
    - extension: str, optional - The file extension of the exported SPSS file. Defaults to "zsav".
    - external_order_path: str, optional - The path to an external order file containing var value labels. Defaults to None.
    - col_threshold: int, optional - The threshold index for separating technical columns from work columns. Defaults to 2.
    - tech_names: list, optional - List of column names for technical variables
    - value_threshold: int, optional - The threshold value for considering a column as a nominal var. Defaults to 12.
    - mp_separator: str, optional - The separator used for multipunch questions. Defaults to " / ".

    Returns:
    None
    """

    def _fill_mp_missing_values(_row, _sub_questions):
        """
        Fill missing values in the specified multipunch _sub_questions of the given _row.

        Parameters:
            _row (pandas.Series): The _row to fill missing values in.
            _sub_questions (list): List of column names representing the _sub_questions.

        Returns:
            pandas.Series: The _row with missing values filled in the specified _sub_questions.
        """
        # Check if any of the _sub_questions have non-missing values
        if _row.loc[_sub_questions].any():
            # Fill missing values with 0
            _row[_sub_questions] = _row[_sub_questions].astype(float).fillna(0)
        return _row

    # Make a copy of the DataFrame and drop columns with all NaN values if empty cols provided
    _df = _df.dropna(axis=1, how='all')

    # Create a _metadata object
    _external_meta = metadata_obj()

    # Separate technical columns from work columns to start var numeration
    _tech_cols, _work_cols = _df.columns[:col_threshold], _df.columns[col_threshold:]
    # treat all tech columns as strings
    for i, col in enumerate(_tech_cols):
        _external_meta.var_names.append(tech_names[i])
        _external_meta.var_names_to_labels[tech_names[i]] = col
        _external_meta.var_measure[tech_names[i]] = 'nominal'
        _external_meta.var_formats[tech_names[i]] = b'A12000'
        _external_meta.var_types[tech_names[i]] = 12000

    # handle external order file
    _ord_var_value_labels = {}
    if external_order_path:
        _external_df = pd.read_excel(external_order_path, index_col=0, header=0, dtype='string')
        _external_df = _external_df.dropna(axis=1, how='all')
        _external_df.index = range(1, len(_external_df) + 1)
        _ord_var_value_labels = _external_df.to_dict()
        _ord_var_value_labels = {question_label:
            {int(str(k).strip()): str(v) for k, v in values.items() if not pd.isna(v)} # keys to int, values to str
                for question_label, values in _ord_var_value_labels.items()}

    # setup for iterations over work columns
    _qst_count = 0
    _mp_set = {}
    _default_na = {0: 'Не выбран'}

    for i, col in enumerate(_work_cols):
        unique_values = unique_no_nan(_df[col])
        # unique_values = _df[col].dropna().unique().tolist()

        if col in _ord_var_value_labels:
            # Check if the column has ordinal var value labels
            input_values = list(_ord_var_value_labels[col].values())
            input_check = all([_df_value in input_values for _df_value in unique_values])

            if input_check:
                # If all unique values in the column are present in the input values, use the provided labels
                temp_var_value_labels = {**_default_na, **_ord_var_value_labels[col]}
            else:
                # Raise an error if the labels in the column are different from the provided labels
                raise ValueError(f'Проверь метки в столбце "{col}" - они отличаются от тех, что в базе')

        elif len(unique_values) < value_threshold:
            # If the number of unique values is less than the threshold, assign unordered value labels
            unordered_value_labels = {i: str(value) for i, value in enumerate(unique_values, start=1)}

            temp_var_value_labels = {**_default_na, **unordered_value_labels}

        else:
            # If the column does not meet the above conditions, it is considered as a OE question
            _qst_count += 1
            name = f"q{str(_qst_count).zfill(4)}_1"
            _external_meta.var_names.append(name)
            _external_meta.var_names_to_labels[name] = col
            _external_meta.var_measure[name] = 'nominal'
            _external_meta.var_formats[name] = b'A12000'
            _external_meta.var_types[name] = 12000
            continue

        # Check if the question is multipunch
        condition_label_length = len(temp_var_value_labels) == 2  # multipunches always have 2 labels



        mp_sep_value_labels = f"{mp_separator}{temp_var_value_labels[1]}" # value 1 label with multipunch separator
        condition_mp_separator = mp_sep_value_labels in col # multipunches always have the value 1 after separator

        if condition_label_length and condition_mp_separator:
            # Extract the body label of the multipunch question
            mp_body_label = col.replace(mp_sep_value_labels, "")

            # Check if it is the first sub-question in multipunch cascade, catching if it is the first question in base
            if i > 1:
                prev_col_label = _work_cols[i - 1]
            else:
                prev_col_label = col

            if not f"{mp_body_label}{mp_separator}" in prev_col_label:
                _qst_count += 1
                sub_counter = 1
            else:
                # catch if the labels has errors
                try:
                    sub_counter += 1
                except:
                    raise ValueError(f'Проверь столбец "{col}" - предыдущий столбец должен быть многоответным')

            mp_body_name = f"q{str(_qst_count).zfill(4)}"
            name = f"{mp_body_name}_{sub_counter}"
            _external_meta.var_names_to_labels[name] = mp_body_label

            # For the first mp encounter create empty list for subquestions
            if _mp_set.get(mp_body_name) is None:
                _mp_set[mp_body_name] = []

            _mp_set[mp_body_name].append(name)

        # Otherwise it is singlepunch question
        else:
            _qst_count += 1
            name = f"q{str(_qst_count).zfill(4)}"
            _external_meta.var_names_to_labels[name] = col

        # Apply result to the _metadata
        _external_meta.var_names.append(name)
        _external_meta.var_value_labels[name] = temp_var_value_labels

    # assign column names to the dataframe
    _df.columns = _external_meta.var_names

    # recode dataframe values in numeric codes for nominal questions
    recode_values = {
        str(out_k): {str(in_v): str(in_k) for in_k, in_v in out_v.items()}
            for out_k, out_v
                in _external_meta.var_value_labels.items()
    }

    recode_cols = list(_external_meta.var_value_labels.keys())
    _df[recode_cols] = _df[recode_cols].astype(str).replace(recode_values).apply(pd.to_numeric, errors='coerce')


    # update other _metadata properties with resulted vars
    _external_meta.update(_df)

    # fill missing values in multipunch questions where applicable
    for mp_sub_set in list(_mp_set.values()):
        _df = _df.apply(_fill_mp_missing_values, axis=1, _sub_questions=mp_sub_set)

    # export resulted dataframe to spss
    write_spss(df=_df,
               meta=_external_meta,
               path_to_export=path_to_export,
               export_file_name=export_file_name,
               extension=extension,
               custom_meta=None
    )

def check_scales_questions(_df, _meta, add_recodes = True):
    # Write the new vars we create here
    _new_vars = []

    # Wording that is out of half scales
    _na_wording = [
        'затрудняюсь ответить',
        'нет ответа',
        'не применимо ко мне',
        'затруднились'
    ]

    # Common pairs of polarizing words for halfscales
    _diff_hlfscl_wrds = [
        # ['не'],
        ['хорош', 'плох'],
        ['положитель', 'отрицатель'],
        ['позитив', 'негатив'],
        ['лучше', 'хуже'],
        ['улучш', 'ухудш'],
        ['полностью', 'совершенно'],
        ['полностью', 'совсем'],
        ['значитель', 'незначитель'],
        ['выше', 'ниже'],
        ['высок', 'низк'],
        ['легк', 'сложн'],
        ['очень', 'совершенн'],
        ['повыс', 'сниз'],
    ]

    def _is_one_step_range(_var_val_labels):
        """
        Check if the provided dictionary of var labels represents a range with a step of one.

        Args:
            _var_val_labels (dict): A dictionary containing var labels as keys and their corresponding values.

        Returns:
            bool: True if the var labels represent a range with a step of one, False otherwise.
        """

        # Get the values from the dictionary and remove any NaN values
        _not_nan_vals = [value for value in _var_val_labels.values() if value is not None]

        # Check if the list is empty or has only one element
        if not _not_nan_vals or len(_not_nan_vals) <= 1:
            return False

        # Check if it's either ascending or descending range starting with 0 or 1
        for start in [0, 1]:
            _is_ascending = _not_nan_vals[0] == start and all(
                _not_nan_vals[i] - _not_nan_vals[i - 1] == 1 for i in range(1, len(_not_nan_vals)))
            if _is_ascending:
                return True

            _is_descending = _not_nan_vals[-1] == start and all(
                _not_nan_vals[i - 1] - _not_nan_vals[i] == 1 for i in range(1, len(_not_nan_vals)))
            if _is_descending:
                return True

        return False

    def _recode_chunks(_not_nan_val_labels, _chunk_size):
        """
        Recode the values in the provided dictionary based on chunk size.

        Args:
            _not_nan_val_labels (dict): A dictionary containing var labels as keys and their corresponding values.
            _chunk_size (int): The size of each chunk for recoding.

        Returns:
            dict: A dictionary with the recoded values based on the chunk size.
        """
        # Extract the keys and values from the dictionary
        _not_nan_keys = list(_not_nan_val_labels.keys())
        _not_nan_values = list(_not_nan_val_labels.values())

        # Iterate over the chunk size
        for i in range(_chunk_size):
            # Replace the top N and bottom N values with recoded labels
            _not_nan_values[i] = f"TOP-{_chunk_size}"
            _not_nan_values[i * (-1) - 1] = f"BOTTOM-{_chunk_size}"

        # Create a recode dictionary using the modified keys and values
        _rec_dict_for_agg = {k: v for k, v in zip(_not_nan_keys, _not_nan_values)}

        return _rec_dict_for_agg

    def beautify_scale_labels(_var_val_labels, _rec_dict_agg):
        """
        Beautify the scale labels by updating them with aggregated recode and transforming the keys.

        Args:
            _var_val_labels (dict): A dictionary containing var labels as keys and their corresponding values.
            _rec_dict_agg (dict): A dictionary containing the recoded values for aggregation.

        Returns:
            tuple: A tuple containing the transformed dictionary and the recoded var value labels.
        """

        # Create recode for aggregate and update with top-bottom recode
        _var_val_labels = dict(_var_val_labels)
        _var_val_labels.update(_rec_dict_agg)

        # Create dict that will recode original values for aggregated recode
        _first_label_tracker = {}
        _rec_into_agg_var = {}

        # beautify keys to be range
        for key, value in _var_val_labels.items():
            if value not in _first_label_tracker:
                _first_label_tracker[value] = key
                _rec_into_agg_var[key] = key
            else:
                _rec_into_agg_var[key] = _first_label_tracker[value]

        # Get the keys and values from the transformed dictionary
        _keys = list(_rec_into_agg_var.keys())
        _values = list(_rec_into_agg_var.values())

        # Fill the gaps in the sequence
        # Iterate over the values because next check should be after change applied
        for i in range(len(_values)):
            if i == 0:
                continue

            if _values[i] - _values[i - 1] > 1:
                _values = [_values[i - 1] + 1 if x == _values[i] else x for x in _values]

        _rec_into_agg_var = dict(zip(_keys, _values))

        # gather unique labels of aggregated values
        _uniq_agg_labels = []
        for i in range(len(_var_val_labels)):
            if _var_val_labels[i] not in _uniq_agg_labels:
                _uniq_agg_labels.append(str(_var_val_labels[i]))

        # Create a new dictionary with the transformed keys and unique labels
        _agg_var_val_labels = dict(zip(_keys, _uniq_agg_labels))

        return _rec_into_agg_var, _agg_var_val_labels

    def _get_text_before_underscore(_string):
        """
        Extracts the text before the last underscore in a given _string.

        Args:
            _string (str): The input _string.

        Returns:
            str or None: The text before the last underscore if it exists, None otherwise.
        """
        # Split the _string by the pattern "_<digits>"
        _parts = re.split(r"_(\d+)$", _string)
        # Check if the split produced more than one part
        if len(_parts) > 1:
            # Return the first part, which is the text before the last underscore
            return _parts[0]
        else:
            # Return None if no underscore is found
            return None

    def _is_last_of_the_block(_var, _idx, _vars):
        """
        Check if a _var is the last one of a block in a list of _vars.

        Parameters:
            _var (any): The _var to check.
            _idx (int): The _idx of the _var in the list.
            _vars (list): The list of _vars.

        Returns:
            bool or str: Returns True if the _var is the last one of the block,
                         or the block name as a string if it is not the last one.
                         Returns False if the _var is not part of a block.
        """
        try:
            next_var = _vars[_idx + 1]
        except:
            # If there is no next _var, then the current _var is the last one
            return True
        _var_body = _get_text_before_underscore(_var)
        _next_var_body = _get_text_before_underscore(next_var)
        if _var_body != _next_var_body:
            # If the current _var and the next _var have different block names,
            # then the current _var is the last one of the block
            return _var_body
        else:
            # If the current _var and the next _var have the same block name,
            # then the current _var is not the last one of the block
            return False


    def _check_dk_scenarios(_dk_val, _rec_vals):
        """
        Check DK scenarios and determine if the values create polarizing pairs based on wording differentiation.
        Parameters:
            _dk_val: the DK value to be excluded
            _rec_vals: a dictionary of values to be checked for polarizing pairs

        Returns:
            A boolean indicating if the values create polarizing pairs and a dictionary of aggregated recoded values
        """

        # Pop NA and DK - we don't need them here
        _rec_vals = {key: value for key, value in _rec_vals.items() if key not in [na_value, _dk_val]}

        # Here we check, if the remaining values create polarising pairs by the wording differentiation
        # There could be uneven number - it's ok, assume the middle number is the real middle :)
        split_value_pares = [(list(_rec_vals.values())[i], list(_rec_vals.values())[-(i + 1)])
                             for i in range(len(_rec_vals) // 2)]

        # list of boolean results of pair polarizing comparison
        pair_halfscale_match = []
        for pair in split_value_pares:
            string_1 = pair[0].lower()
            string_2 = pair[1].lower()

            # Generate the differences using difflib
            diff = difflib.ndiff(string_1.split(), string_2.split())

            # Extract the differing words, difflib provides stupid + and - prefixes
            difflib_stupid_prefix = 2
            differing_words = [word[difflib_stupid_prefix:] for word in diff if
                               word.startswith('+ ') or word.startswith('- ')]

            # If the denying word "не" is in difference and up to 4 other words (assumption), this is it
            threshold = 4

            # check for the different wording condition
            result = False
            if (len(differing_words) <= threshold):
                # if there is match on predefined polarizing wording
                cond_diff_match = any([
                        all(
                            any(s in diff_s for diff_s in differing_words) # predefined word in any diff word
                            for s in differentiating_halfscales_words_pair) # all predefined words from pair in var pair
                        for differentiating_halfscales_words_pair in _diff_hlfscl_wrds])

                # if the difference is just NO word
                cond_only_no = (any(['не' in diff_s for diff_s in differing_words]) and (len(differing_words) == 1))

                # if the NO is merged and the difference is 2 NO divided words
                cond_no_inside = (any([diff_s.startswith('не') for diff_s in differing_words]) and
                                  (len(differing_words) == 2))

                if any([cond_diff_match, cond_only_no, cond_no_inside]):
                    result = True

            pair_halfscale_match.append(result)

        # I don't think it's possible, that chunk size is less than 2, but just in case
        chunk_size = len(pair_halfscale_match)
        halfscale_condition = all(pair_halfscale_match) and (chunk_size > 1)
        if halfscale_condition:
            recode_dict_aggregate = _recode_chunks(_rec_vals, chunk_size)
            _meta.half_scale_vars.append(name)
            is_half_scale = True
        else:
            recode_dict_aggregate = None
            is_half_scale = False

        return is_half_scale, recode_dict_aggregate

    # iterate through each numeric var
    for name in _meta.var_value_labels:
        # Flag for awareness variables
        is_scale = False
        is_awareness = False
        is_half_scale = False

        # Select singlepunch vars only
        if len(_meta.var_value_labels.get(name)) < 3:
            continue

        # Check if it is pure scale - var has range as labels
        # Strip to numeric values
        int_values_extracted = {
            k: int(re.match(r'^\d+', str(v)).group()) if re.match(r'^\d+', str(v)) else np.nan
            for k, v in _meta.var_value_labels.get(name).items()
        }

        # Take resulted values and drop NaN
        not_nan_value_labels = {k: v for k, v in int_values_extracted.items() if pd.notna(v)}

        # If labels form perfect range, split them in 2 chunks
        if _is_one_step_range(not_nan_value_labels):
            chunk_size = len(not_nan_value_labels) // 2
            recode_dict_aggregate = _recode_chunks(not_nan_value_labels, chunk_size)
            _meta.scale_vars.append(name)  # Save the name, it will be used for mean tables
            _meta.scale_recode_for_mean[name] = int_values_extracted  # Create recode dict for mean tables
            is_scale = True

        # Check for the awareness variable
        if (not is_scale) and len(_meta.var_value_labels.get(name).values()) == 4:
            if (_meta.var_value_labels.get(name).get(1, "").lower().strip() == 'знаю хорошо') and \
            (_meta.var_value_labels.get(name).get(3, "").lower().strip() == 'слышу впервые'):
                recode_awareness = [{1:1, 2:1, 3:2}, {0: 'Не выбран', 1: "Знаю", 2: "Не знаю"}]
                _meta.awareness_vars.append(name)
                is_awareness = True

        # Check if the question is half scale
        if (not is_scale) and (not is_awareness):
            # create copy to avoid inplace
            recode_values = dict(_meta.var_value_labels.get(name))

            # Try to drop 0 value ("Not chosen")
            try:
                del recode_values[0]
            except:
                pass

            # It seems to be, half scales has from 4 to 6 active values
            if not 4 <= len(recode_values) <= 6:
                continue

            # Split in pieces and find length
            value_keys = list(recode_values.keys())
            value_labels = list(recode_values.values())
            len_value_labels = len(value_labels)

            # Check if there is NA value (use predefined list)
            na_ratios = [max([fuzz.ratio(str(value_label).lower(), string) for string in _na_wording])
                         for value_label in value_labels]
            max_na_ratio = max(na_ratios)
            if max_na_ratio > 90:
                na_index = na_ratios.index(max_na_ratio)
                na_value = value_keys[na_index]
            else:
                na_value = None

            # Check if there is DK value in remaining values
            if na_value:
                # If there is any values after NA
                if na_value < len_value_labels:
                    # If there is more than one value, just drop it - it shouldn't be right
                    if len_value_labels - na_value > 1:
                        continue
                    # Otherwise it's DK
                    middle_dk_value = value_keys[na_index + 1]
                    end_dk_value = value_keys[na_index + 1]
                    # dk_value = value_keys[na_index + 1]

                # If NA is the last value and there is uneven number of values, the last is DK (assumption)
                elif (len_value_labels - 1) % 2 == 1:
                    # from here 2 scenarios should be checked
                    # at first check in middle, if not, check in the end
                    middle_dk_value = value_keys[(len_value_labels-1) // 2]
                    end_dk_value = value_keys[na_index - 1]
                    # dk_value = value_keys[na_index - 1]
                else:
                    middle_dk_value = None
                    end_dk_value = None

                    # dk_value = None
            # If there is no NA, then DK is the last in case of uneven number (assumption)
            else:
                if len_value_labels // 2 == 1:
                    middle_dk_value = value_keys[len_value_labels // 2]
                    end_dk_value = value_keys[-1]
                    # dk_value = value_keys[-1]
                else:
                    middle_dk_value = None
                    end_dk_value = None
                    # dk_value = None

            # At first try to check if neutral is in middle, else in the end
            is_half_scale, recode_dict_aggregate = _check_dk_scenarios(middle_dk_value, recode_values)
            if not is_half_scale:
                is_half_scale, recode_dict_aggregate = _check_dk_scenarios(end_dk_value, recode_values)
                if not is_half_scale:
                    continue


        # If var is any type of scale, create recode var with "rcd_" prefix
        if add_recodes and any([is_scale, is_half_scale, is_awareness]):
            recode_name = f"rcd_{name}"
            _meta.scale_recode_for_aggregate[name], _meta.scale_value_labels[recode_name] = recode_awareness \
                if is_awareness else beautify_scale_labels(
                    _meta.var_value_labels[name], recode_dict_aggregate)
            _df[recode_name] = _df[name].replace(_meta.scale_recode_for_aggregate[name])
            _new_vars.append(recode_name)

    # Wrap it up!

    # The trick is to place recodes halfscales near the original vars
    # And grid scales in one block after the original block based on assumption that grid have "_" in name
    if add_recodes:
        new_columns_order = []

        for i, orig_var in enumerate(_meta.original_vars):
            new_columns_order.append(orig_var)
            recode_name = "rcd_" + orig_var
            if recode_name in _new_vars:
                if re.match(r".*_(\d+)$", orig_var) is None:
                    new_columns_order.append(recode_name)
                    _meta.recode_pairs[orig_var] = recode_name
                else:
                    var_body = _is_last_of_the_block(orig_var, i, _meta.original_vars)
                    if var_body:
                        block_to_append = [var for var in _new_vars if
                                           _get_text_before_underscore(var) == f"rcd_{var_body}"]
                        new_columns_order.extend(block_to_append)
                        _meta.recode_pairs.update({rcd_var[4:]: rcd_var for rcd_var in block_to_append})

        _df = _df[new_columns_order]

        _meta.var_names = list(_df.columns)
        _meta.var_names_to_labels.update(
            {var: f"Рекод: {_meta.var_names_to_labels[var[4:]]}" for var in _new_vars})
        _meta.var_value_labels.update(_meta.scale_value_labels)
        _meta.var_measure.update({var: 'nominal' for var in _new_vars})
        _meta.var_formats.update({var: b'F8.2' for var in _new_vars})
        _meta.var_types.update({var: 0 for var in _new_vars})
        _meta.q_type.update({var: _meta.q_type[var[2:]] for var in _new_vars})
        _meta.var_alignments.update({var: 'center' for var in _new_vars})

    return _df, _meta


def fill_recode_zeros(df, _meta):
    """
    Fills the zero values in the specified variables of a DataFrame with the value 0.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be modified.
        _meta (Meta): The _metadata object containing information about the zero variables.

    Returns:
        Tuple[pandas.DataFrame, Meta]: The modified DataFrame and the updated _metadata object.
    """
    _meta.fill_zero_labels()
    zero_vars = _meta.zero_vars
    df.loc[:, zero_vars] = df.loc[:, zero_vars].fillna(0)
    return df, _meta


def recode_extra_vars(df, _meta):
    """
    Recodes extra variables in the given DataFrame based on the provided _metadata.

    Args:
        df (pandas.DataFrame): The DataFrame to recode.
        _meta (Metadata): The _metadata object containing variable names and labels.

    Returns:
        tuple: A tuple containing the recoded DataFrame and the updated _metadata object.
    """

    banner_extra = {}
    extra_vars = []

    for var, label in _meta.var_names_to_labels.items():
        # Check if the label corresponds to the age question
        if 'укажите ваш возраст' in label.lower():
            if _meta.var_value_labels.get(var, None) is not None:
                # Check if 'менее 18 лет' option exists and isn't in the DataFrame
                if 'менее 18' in  _meta.var_value_labels.get(var).get(1, '_').lower():
                    if 1 not in df[var].values:
                        del _meta.var_value_labels[var][1]
                        age_var = var
                        age_index = list(df.columns).index(age_var)
                        age_recode_name = f'rcd_{age_var}'
                        age_recode = df[age_var].replace({2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}).rename(age_recode_name)
                        df = pd.concat([df.iloc[:, :age_index + 1], age_recode, df.iloc[:, age_index + 1:]], axis=1)

                        _meta.var_names.append(age_recode_name)
                        _meta.original_vars.append(age_recode_name)
                        _meta.var_names_to_labels[age_recode_name] = f'Рекод: {_meta.var_names_to_labels[age_var]}'
                        _meta.var_value_labels[age_recode_name] = {0: 'Не выбран', 1: '18-34 лет', 2: '35-54 лет',
                                                                  3: '55 и более лет'}
                        _meta.var_measure[age_recode_name] = 'nominal'
                        _meta.var_formats[age_recode_name] = b'F8.2'
                        _meta.var_types[age_recode_name] = 0
                        _meta.var_display_widths[age_recode_name] = 10
                        _meta.var_alignments[age_recode_name] = 'center'
                        _meta.q_type[age_recode_name] = 'singlepunch'

                        banner_extra[age_var] = age_recode_name
                        extra_vars.append(age_recode_name)

                        break

    for var, label in _meta.var_names_to_labels.items():
        if 'в каком населенном пункте вы проживаете' in label.lower():
            if _meta.var_value_labels.get(var, None) is not None:
                if len(_meta.var_value_labels.get(var)) == 8:
                    geo_var = var
                    geo_index = list(df.columns).index(geo_var)
                    geo_1_recode_name = f'rcd_{geo_var}_1'
                    geo_2_recode_name = f'rcd_{geo_var}_2'

                    _meta.original_vars.append(geo_1_recode_name)
                    _meta.original_vars.append(geo_2_recode_name)

                    geo_1_recode = df[geo_var].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4}).rename(
                        geo_1_recode_name)
                    geo_2_recode = df[geo_var].replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 4}).rename(
                        geo_2_recode_name)

                    df = pd.concat([df.iloc[:, :geo_index + 1], geo_1_recode, geo_2_recode, df.iloc[:, geo_index + 1:]],
                                   axis=1)

                    _meta.var_names.extend([geo_1_recode_name, geo_2_recode_name])

                    _meta.var_names_to_labels[geo_1_recode_name] = f'Рекод: {_meta.var_names_to_labels[geo_var]}'
                    _meta.var_names_to_labels[geo_2_recode_name] = f'Рекод: {_meta.var_names_to_labels[geo_var]}'

                    _meta.var_value_labels[geo_1_recode_name] = {0: 'Не выбран', 1: 'Москва и СПб', 2: 'Миллионники',
                                                                3: 'От 100 тыс до 1 млн', 4: 'До 100 тыс'}
                    _meta.var_value_labels[geo_2_recode_name] = {0: 'Не выбран', 1: 'От 1 млн чел',
                                                                2: 'От 100 тыс до 1 млн чел', 3: 'До 100 тыс',
                                                                4: 'Село'}

                    _meta.var_measure[geo_1_recode_name] = 'nominal'
                    _meta.var_measure[geo_2_recode_name] = 'nominal'

                    _meta.var_formats[geo_1_recode_name] = b'F8.2'
                    _meta.var_formats[geo_2_recode_name] = b'F8.2'

                    _meta.var_types[geo_1_recode_name] = 0
                    _meta.var_types[geo_2_recode_name] = 0

                    _meta.var_display_widths[geo_1_recode_name] = 10
                    _meta.var_display_widths[geo_2_recode_name] = 10

                    _meta.var_alignments[geo_1_recode_name] = 'center'
                    _meta.var_alignments[geo_2_recode_name] = 'center'

                    _meta.q_type[geo_1_recode_name] = 'singlepunch'
                    _meta.q_type[geo_2_recode_name] = 'singlepunch'

                    banner_extra[geo_var] = [geo_1_recode_name, geo_2_recode_name]

                    extra_vars.extend([geo_1_recode_name, geo_2_recode_name])

                    break

    return df, _meta, banner_extra, extra_vars


def gen_script_dir():
    return os.getcwd()
