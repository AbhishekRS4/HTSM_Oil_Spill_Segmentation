import csv
import json


def write_dict_to_json(file_json, dict_data):
    """
    ---------
    Arguments
    ---------
    file_json : str
        full path of json file to be saved
    dict_data : dict
        dictionary of params to be saved in the json file
    """
    with open(file_json, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict_data, indent=4))
    return


def load_dict_from_json(file_json):
    """
    ---------
    Arguments
    ---------
    file_json : str
        full path of json file to be saved

    -------
    Returns
    -------
    dict_data : dict
        dictionary of params loaded from the json file
    """
    dict_data = {}
    with open(file_json) as fh:
        dict_data = json.load(fh)
    return dict_data


class CSVWriter:
    """
    CSVWriter class for writing tabular data to a csv file

    ----------
    Attributes
    ----------
    file_name : str
        file name of the csv file
    column_names : list
        a list of column names

    """

    def __init__(self, file_name, column_names):
        self.file_name = file_name
        self.column_names = column_names

        self.file_handle = open(self.file_name, "w")
        self.writer = csv.writer(self.file_handle)

        self.write_header()
        print(f"{self.file_name} created successfully with header row")

    def write_header(self):
        """
        writes header into csv file
        """
        self.write_row(self.column_names)
        return

    def write_row(self, row):
        """
        writes a row into csv file

        ---------
        Arguments
        ---------
        row : list
            a list of row values
        """
        self.writer.writerow(row)
        return

    def close(self):
        """
        close the file
        """
        self.file_handle.close()
        return
