currentUserType = None  # None: 无用户, 1: 管理员, 0: 普通用户
currentUserName = None
FIXED_KEY = b'1AEZy_Ufi7J6WlYdSO4hHbPbbA_ZDlZ3Gh_gICx9J9Q='


import os


class Config:
    # Secret key for session and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY', '088f2304561426d6c0ff97af74ba03aa42c0c3e5a8995396bfd148bb6dd3cbd2')

    # MySQL database URI (use your actual credentials)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:root@localhost/class_emotion')

    # Optional: Enable SQLAlchemy's echoing of SQL statements for debugging
    SQLALCHEMY_ECHO = os.environ.get('SQLALCHEMY_ECHO', False)  # Set to True for SQL logging


    names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Focus', 'Sad', 'Surprised']
    colors = ['#FF0000', '#800080', '#000000', '#00FF00', '#0000FF', '#000080', '#FFFF00']

