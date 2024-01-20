def f_grepl(pattern, l_str):

    """
    Searches for matches to argument 'pattern' within each element of a character list, 'l_str'.
    The Python equivalent of 'grepl' in R.

    Parameters
    ----------
    pattern : 'str'
        Regex pattern.
    l_str : 'list'
        Character list.

    Returns
    -------
    list
        Boolean list, True in case of a match and False in case of a non-match.

    Testing
    -------
    f_grepl("^P", ["Pieter", "Bart", "Theo", "aPieter"])
    """

    return [bool(re.search(pattern, x)) for x in l_str]
