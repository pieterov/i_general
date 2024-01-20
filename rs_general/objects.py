"""
Purpose:    Define objects used at general level
Author:     Pieter Overdevest
"""

# Path to code folder of this project.
C_PATH_CODE = os.getcwd()

# Paths to project root folder and sub-folders.
C_PATH_PROJECT_ROOT = re.sub("Code", "", C_PATH_CODE)
C_PATH_DATA         = os.path.join(C_PATH_PROJECT_ROOT, "Data")
C_PATH_DELIVERABLES = os.path.join(C_PATH_PROJECT_ROOT, "Deliverables")
C_PATH_DOCUMENTS    = os.path.join(C_PATH_PROJECT_ROOT, "Documents")
C_PATH_IMAGES       = os.path.join(C_PATH_PROJECT_ROOT, "Images")
