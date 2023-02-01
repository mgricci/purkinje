import os

def ensure_dir(path: str) -> str:
    ''' Creates the directories specified by path if they do not already exist.

    Parameters:
        path (str): path to directory that should be created

    Returns:
        return_path (str): path to the directory that now exists
    '''
    path = os.path.abspath(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            # if the exception is raised because the directory already exits,
            # than our work is done and everything is OK, otherwise re-raise the error
            # THIS CAN OCCUR FROM A POSSIBLE RACE CONDITION!!!
            if exception.errno != errno.EEXIST:
                raise
    return path

