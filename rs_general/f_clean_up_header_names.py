def f_clean_up_header_names(l_input):

    """
    Clean up header names of data frame: (1) set names to lower case, and (2) replace spaces by '_'.

    Parameters
    ----------
    l_input : list
        Column names.


    Returns
    -------
    list
        Cleaned up column names.
    """
    

    return [
        # Put in lower case:
        x3.lower() for x3 in [

        # Replace space by '_':
        x2 if f_is_numerical(x2) else re.sub(" |\.", "_", x2) for x2 in [

        str(x1) for x1 in l_input        
    ]]]


#######################################################################################################################

def f_check_nonnumeric_in_df(
    
    df_input,
    l_exclude_columns = [],
    l_include_columns = []
    ):

    """
    Check on non-numeric in data frame.
    """

    """
    <short description>.

    Parameters
    ----------
    <name> : <type>
        <short description>.
    <name> : <type>
        <short description>.

    Returns
    -------
    <type>
        <short description>.

    Testing
    -------
    df_input          = df_nline_w_source
    l_exclude_columns = ['product']
    l_include_columns = []
    """

        
    # Error check - Are all column names in 'l_exclude_column' and 'l_include_column' present in df_input?
    l_exclude_columns_not_in_df_input = [x for x in l_exclude_columns if x not in  df_input.columns]
    l_include_columns_not_in_df_input = [x for x in l_include_columns if x not in  df_input.columns]

    if len(l_exclude_columns_not_in_df_input) > 0:

        c_temp = ', '.join(f"'{x}'" for x in l_exclude_columns_not_in_df_input)
        
        raise ValueError(
            f"The following column name(s) in 'l_exclude_columns' are not present in the column names of 'df_input': {c_temp}"
        )

    if len(l_include_columns_not_in_df_input) > 0:

        c_temp = ', '.join(f"'{x}'" for x in l_include_columns_not_in_df_input)
        
        raise ValueError(
            f"The following column name(s) in 'l_include_columns' are not present in the column names of 'df_input': {c_temp}"
        )


    # Error check - Are the same column names present in both 'l_exclude_column' and 'l_include_column'?
    l_overlap_include_exclude_columns = set(l_include_columns).intersection(set(l_exclude_columns))

    if len(l_overlap_include_exclude_columns) > 0:

        c_temp = ', '.join(f"'{x}'" for x in l_overlap_include_exclude_columns)
        
        raise ValueError(
            f"The following column name(s) are present in both 'l_exclude_columns' and 'l_include_columns': {c_temp}"
        )


    # Initialization.
    if l_include_columns == []:
        l_include_columns = df_input.columns


    # Main.
    df_to_check = df_input[[x for x in df_input.columns if x in l_include_columns and x not in l_exclude_columns]]
    df_eval     = df_to_check[~df_to_check.applymap(np.isreal).all(axis = 1)]

    if df_eval.shape[0] > 0:

        print(
            f"\nWARNING - '{f_var_name(df_input)}' contains non-numerical. We observe {df_eval.shape[0]} row(s) with at "
            f"least one non-numerical, below we show the first 5 rows (at max):\n"
        )

        print(df_eval.head(5))

        print("\nFor reference, the full data frame, incl. those columns that were not evaluated:\n")

        print(df_input.filter(items = df_eval.index, axis=0).head(5))

        print("\n")

    else:

        print(f"\nOK - '{f_var_name(df_input)}' contains numericals only.\n")
