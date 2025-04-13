import pandas as pd


def load_csv_file(file_path):
    """
    Load raw data from a CSV file and handle common errors gracefully.
    """

    if not file_path.lower().endswith(".csv"):
        raise ValueError(f"'{file_path}' is not a CSV file.")

    try:
        data = pd.read_csv(file_path)

        if data.empty:
            raise ValueError(f"'{file_path}' is empty.")

        return data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e

    except PermissionError as e:
        raise PermissionError(f"Permission denied for : {file_path}") from e

    except pd.errors.EmptyDataError as e:
        raise ValueError(f"'{file_path}' is empty.") from e

    except pd.errors.ParserError as e:
        raise ValueError(f"'{file_path}' not a properly formatted CSV.") from e

    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1,
            f"Could not decode file '{file_path}'."
        ) from e

    except Exception as e:
        raise RuntimeError(f"Error while reading '{file_path}': {e}") from e
