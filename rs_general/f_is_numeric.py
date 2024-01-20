# Test whether number is number
def f_is_numerical(value):

    try:
        float(value)
        return True
    
    except ValueError:
        return False
     