def f_read_data_from_file(

    l_name,
    c_path,
    c_type               = 'xlsx',
    c_sheet              = None,
    c_sep                = ',',
    l_cols               = None,
    n_skiprows           = None,
    n_rows               = None,
    n_header             = 0,
    b_clean_header_names = True,
    b_strip_spaces       = True
    ):

    """
    Read data from file into a data frame object.

    Parameters
    ----------   
    l_name: 'str'
        List of (parts of) of file names where data is to be read from.
    c_path: 'str'
        Path where file resides.
    c_type: 'str'
        Reference to file type to be read (default: 'xlsx').
    c_sheet: 'str'
        Sheet name in case data is to read from Excel file (default: 'None').
    l_cols: 'int',
        If list of int, then indicates list of column numbers to be parsed (0-indexed).
    n_skiprows: 'int'
        Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. 
    n_rows: 'int'
        Number of rows to parse (default: None).
    n_header: 'int'
        Row (0-indexed) to use for the column labels of the parsed DataFrame.
    b_clean_header_names: 'bool'
        Do we clean up the header names? (default: True)
    b_strip_spaces: 'bool'
        Do we strip spaces before and after the data in each cell? (default: False)

    Returns
    -------
    Data frame
        Data read from file is stored in data frame object.

    Testing
    -------  
    Default:
    c_type               = 'xlsx'
    c_sheet              = None
    c_sep                = ','
    l_cols               = None
    n_skiprows           = None
    n_rows               = None
    n_header             = 0
    b_clean_header_names = True
    b_strip_spaces       = True


    l_name  = ["Classification GIs - " + x for x in l_comp_list]
    c_path  = C_PATH_DATA
    

    l_name  = 'Content Database'
    c_path  = C_PATH_DATA
    c_type  = 'xlsx'
    c_sheet = 'gedragsindicatoren'

    """ 


#----------------------------------------------------------------------------------------------------------------------
# Initialization.
#----------------------------------------------------------------------------------------------------------------------

    # Valid file types.
    l_type_valid = ['xlsx', 'xlsm', 'csv', 'parquet']

    # Convert l_name to list - with string as single element - in case it is a string.
    if isinstance(l_name, str):
        l_name = [l_name]

    # Latest file per file name. c_name='Content Database'
    l_file = [

        f_get_latest_file(c_name, c_path, c_type)

        for c_name in l_name
    ]

    # Create empty list
    l_df_data = []


#----------------------------------------------------------------------------------------------------------------------
# Error check.
#----------------------------------------------------------------------------------------------------------------------

    if c_type not in l_type_valid :
        raise ValueError(f"You did not provide a valid file type. Choose 'c_type' to be one of {', '.join(l_type_valid)}.")


#----------------------------------------------------------------------------------------------------------------------
# Main.
#----------------------------------------------------------------------------------------------------------------------

    # Iterate through all file names.
    for i in range(len(l_name)): # i=0

        # Excel
        if c_type in ['xlsx', 'xlsm']:            
                
            obj = pd.read_excel(

                io         = os.path.join(c_path, l_file[i].file),
                sheet_name = c_sheet,
                usecols    = l_cols,
                skiprows   = n_skiprows,
                nrows      = n_rows,
                header     = n_header,
                engine     = 'openpyxl'
            )

            # If we don't supply a sheet name the output is a dictionary of data frames,
            # from which we will take the first worksheet.
            if isinstance(obj, dict):
                    
                c_sheet_temp = list(obj.keys())[0]
                
                df_temp = obj[c_sheet_temp]

            # When sheet name does exist the output is a data frame.
            else:

                c_sheet_temp = None

                df_temp = obj


            # Append data frame to list of data frames.
            l_df_data.append(df_temp)


        # CSV
        if c_type == 'csv':

            l_df_data.append(
                
                pd.read_csv(

                    filepath_or_buffer = os.path.join(c_path, l_file[i].file),
                    sep                = c_sep,
                    usecols            = l_cols,
                    skiprows           = n_skiprows,
                    header             = n_header
                )
            )


        # Parquet
        if c_type == 'parquet':

                l_df_data.append(

                    pd.read_parquet(

                        path    = os.path.join(c_path, l_file[i].file),
                        engine  = 'pyarrow',
                        columns = l_cols
                    )
                )

       

        # Comms to the user.
        print(f"\nReading at : {datetime.now()}")

        print(f"Requested  : '{l_name[i]}' (file), '{c_type}' (type)")

        print(f"Read file  : '{l_file[i].file}'")

        if c_type in ['xlsx', 'xlsm']:
            
            if c_sheet is not None:
                
                print(f"Sheet name : '{c_sheet}' - provided by you.")

            else:

                print(f"Sheet name : '{c_sheet_temp}' - first sheet in the workbook.")


        print(f"Path       : '.../{re.sub(f_who_am_i()[1], '', c_path)}'")

        print(f"Modified   : {l_file[i].date_mod}")

        print(f"Age        : {l_file[i].age}")

        print(f"==========================")


    # Concatenate data frames.
    df_data = pd.concat(l_df_data)

    # Clean up header names?
    if b_clean_header_names:
        df_data.columns = f_clean_up_header_names(l_input = df_data.columns)


    # Strip spaces before and after the data in each cell?
    if b_strip_spaces:
        df_data = df_data.map(lambda x: x.strip() if isinstance(x, str) else x)


#----------------------------------------------------------------------------------------------------------------------
# Return results.
#----------------------------------------------------------------------------------------------------------------------

    return df_data
