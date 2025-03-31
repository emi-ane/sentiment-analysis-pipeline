import pandas as pd


def load_csv_file(file_path):
    """Load raw data from a CSV file while handling errors gracefully."""

    if not file_path.lower().endswith(".csv"):
        raise ValueError(f"Error: The file '{file_path}' is not a CSV file.")

    try:
        data = pd.read_csv(file_path)

        if data.empty:
            raise ValueError(f"Error: The file '{file_path}' is empty.")

        return data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.") from e
    except PermissionError as e:
        raise PermissionError(
            f"Error: Permission denied when trying to read the file '{file_path}'."
        ) from e
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Error: The file '{file_path}' is empty.") from e
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Error: The file '{file_path}' is incorrectly formatted."
        ) from e
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            f"Error: The file '{file_path}' could not be decoded. Check the file encoding."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while processing '{file_path}'. Details: {e}"
        ) from e
