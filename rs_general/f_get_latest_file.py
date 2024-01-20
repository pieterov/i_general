def f_get_latest_file(

    c_name,
    c_path,
    c_type
    ):

    """
    Get latest file with said string in the file residing in said path.

    Parameters
    ----------   
    c_name: 'str'
        String in the file name.
    c_path: 'str'
        Path where file resides.
    c_type: 'str'
        Reference to file type to be read.

    Returns
    -------
    Pandas Series
        file: file name
        date_mod: modification date
        age: age of file as string

    Testing
    -------  
    c_name = 'HTRI'
    c_path = C_PATH_DELIVERABLES
    c_type = 'xlsx'

    c_name = 'Den Haag - Displays - Overzicht na plaatsing'
    c_path = c_path_file_xls
    c_type = 'xlsx'

    f_get_latest_file(c_name, c_path, c_type)
    """ 


#----------------------------------------------------------------------------------------------------------------------
# Initialization.
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------
# Error check.
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------
# Main.
#----------------------------------------------------------------------------------------------------------------------

    # Get all files in said folder, excl. any folders. I replaced f_find_str by list(filter(re.compile(c_name).search, list))
    df_file = pd.DataFrame({
        'file': list(filter(
            
            # String to search for in the file names.
            re.compile(c_name).search,

            # List with all files in c_path.
            [                
                f for f in os.listdir(c_path)

                # Filter on files only (excl dirs) and on the requested file type.
                if os.path.isfile(os.path.join(c_path, f)) and
                    os.path.splitext(os.path.join(c_path, f))[1]== '.'+c_type
            ]
        ))
    })

    # Error check - Is a file found?
    if df_file.shape[0] == 0:
        raise LookupError(
            f"No file found for:\nFile name: '{c_name}'\nFile type: '{c_type}'\nFile path: '{c_path}'"
        )

    # Add number of seconds since epoch.
    df_file.insert(1, 'date_mod_sec',
        [os.path.getmtime(os.path.join(c_path, f)) for f in df_file.file]
    )
    
    # Get first row (latest file).
    ps_file = (df_file
        .sort_values(by='date_mod_sec', ascending=False)
        .iloc[0]
    )

    # Convert seconds to time stamp.
    ps_file['date_mod'] = datetime.fromtimestamp(
        
        ps_file.date_mod_sec
        
        ).strftime('%Y-%m-%d %H:%M:%S')
  
    # Add age of file
    n_age = time.time() - ps_file.date_mod_sec

    if n_age < 60:
        c_age = 'sec'
    elif n_age < 3600:
        n_age = n_age / 60
        c_age = 'minutes'
    elif n_age < 3600*24:
        n_age = n_age / 3600
        c_age = 'hours'
    else:
        n_age = n_age / 3600 / 24
        c_age = 'days'

    ps_file['age'] = str(round(n_age,1)) + " " + c_age


#----------------------------------------------------------------------------------------------------------------------
# Return results.
#----------------------------------------------------------------------------------------------------------------------

    # Return pandas series with information, except 'date_mod_sec' (not needed).
    return ps_file.drop('date_mod_sec')
