# init_db.py
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


# 数据库配置类
class Config:
    # Secret key for session and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY', '088f2304561426d6c0ff97af74ba03aa42c0c3e5a8995396bfd148bb6dd3cbd2')

    # MySQL database URI (use your actual credentials)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:root@localhost/class_emotion')

    # Optional: Enable SQLAlchemy's echoing of SQL statements for debugging
    SQLALCHEMY_ECHO = os.environ.get('SQLALCHEMY_ECHO', False)  # Set to True for SQL logging


# 创建数据库引擎
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=Config.SQLALCHEMY_ECHO)

# 创建会话工厂
Session = sessionmaker(bind=engine)

# 创建基类
Base = declarative_base()


# 定义模型
class User(Base):
    __tablename__ = 'user_info'

    id = Column(Integer, primary_key=True)
    user_name = Column(String(80), unique=True, nullable=False)
    user_password = Column(String(80), unique=False, nullable=False)
    gender = Column(String(120), unique=False, nullable=True)
    des = Column(String(120), unique=False, nullable=True)
    age = Column(String(120), unique=False, nullable=True)
    account = Column(String(80), unique=False, nullable=True)
    delt = Column(Integer, default=0, server_default="0", nullable=False)  # 双重保险


# 初始化数据库并创建表
def init_db():
    # 创建所有表
    Base.metadata.create_all(engine)
    print("数据库初始化成功！")


# 运行数据库初始化函数
if __name__ == '__main__':
    init_db()