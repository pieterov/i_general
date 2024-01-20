def f_get_root_folder():

    """
    Get root folder of this computer.

    Parameters
    ----------
    -

    Returns
    -------
    str
        Root folder.
    """
        
    # Root folder (Partner folder).
    return re.search(r'.+/Partners/', os.getcwd()).group()