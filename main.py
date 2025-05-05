import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QStackedWidget
from PyQt5 import QtCore
from register import RegisterPage as registerPageClass
from LogIn import LogInPage as logInPageClass
import pymysql
from pymysql.cursors import DictCursor
from version1 import EmotionMonitoringUI
from PyQt5.QtCore import pyqtSignal


import background


class DbManager:
    def __init__(self, db_config):
        self.connection = pymysql.connect(**db_config, cursorclass=DictCursor)
        self.cursor = self.connection.cursor()
        print("数据库连接成功")

    def insert_data(self, table_name, data):
        # 构造SQL语句和参数
        fields = ', '.join(data.keys())
        values = tuple(data.values())
        placeholders = ', '.join(['%s'] * len(data))
        sql = f"INSERT INTO {table_name} ({fields}) VALUES ({placeholders})"
        try:
            self.cursor.execute(sql, values)
            self.connection.commit()
            return True
        except pymysql.MySQLError as e:
            print(f"向表 {table_name} 插入数据失败: {e}")
            return False

    def query_data_raw(self, query, params=None):
        """
        执行原生 SQL 查询，支持参数化查询
        :param query: SQL 查询语句
        :param params: 查询参数（可选）
        :return: 查询结果列表，失败时返回 None
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except pymysql.MySQLError as e:
            print(f"数据库查询失败：{e}")
            return None

    def query_userinfo(self, table_name, query_condition, query_value):
        sql = f"SELECT * FROM {table_name} WHERE {query_condition} = %s"
        try:
            self.cursor.execute(sql, (query_value,))
            results = self.cursor.fetchall()  # 获取所有查询结果
            return results
        except pymysql.MySQLError as e:
            print(f"查询数据失败: {e}")
            return None

    def update_userinfo(self, table_name, query_condition, query_value, update_data):
        # 查询当前表中的数据
        current_data = self.query_userinfo(table_name, query_condition, query_value)

        if not current_data:
            print(f"未找到匹配的记录: {query_condition} = {query_value}")
            return False

        current_data = current_data[0]  # 假设只有一条记录

        # 比较新数据和旧数据
        update_fields = []
        update_values = []
        for key, value in update_data.items():
            if current_data.get(key) != value:
                update_fields.append(f"{key} = %s")
                update_values.append(value)

        if not update_fields:
            print("没有需要更新的数据")
            return False

        # 构建更新语句
        update_fields_str = ", ".join(update_fields)
        update_values.append(query_value)
        sql = f"UPDATE {table_name} SET {update_fields_str} WHERE {query_condition} = %s"

        try:
            self.cursor.execute(sql, update_values)
            self.connection.commit()
            print("数据更新成功")
            return True
        except pymysql.MySQLError as e:
            print(f"数据更新失败: {e}")
            self.connection.rollback()
            return False

    def update_data(self, table, set_field, set_value, condition_field, condition_value):
        """
        更新数据库中的记录

        :param table: 表名
        :param set_field: 要更新的字段名
        :param set_value: 要更新的字段值
        :param condition_field: 条件字段名
        :param condition_value: 条件字段值
        :return: 更新是否成功
        """
        try:
            cursor = self.connection.cursor()
            query = f"""
                UPDATE {table}
                SET {set_field} = %s
                WHERE {condition_field} = %s
            """
            cursor.execute(query, (set_value, condition_value))
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"更新失败：{str(e)}")
            return False

    def update_user_role(self, table_name, query_condition, query_value, new_role):
        sql = f"UPDATE {table_name} SET user_role = %s WHERE {query_condition} = %s"
        try:
            self.cursor.execute(sql, (new_role, query_value))
            self.connection.commit()
            return True
        except pymysql.MySQLError as e:
            print(f"更新角色失败: {e}")
            self.connection.rollback()
            return False

    def query_user_info(self, user_name):
        table_name = "user_info"
        query_condition = "user_name"
        delt_condition = "delt"

        # 构建 SQL 查询语句
        sql = f"SELECT * FROM {table_name} WHERE {query_condition} = %s AND {delt_condition} = 0"

        try:
            self.cursor.execute(sql, (user_name,))
            results = self.cursor.fetchall()  # 获取所有查询结果
            return results
        except pymysql.MySQLError as e:
            print(f"查询数据失败: {e}")
            return None

    def update_user_data(self, user_name, new_data):
        """
        更新用户信息
        :param user_name: 要更新的用户名
        :param new_data: 包含新数据的字典，例如 {"age": 30, "address": "New York"}
        :return: True 表示更新成功，False 表示更新失败
        """
        table_name = "user_info"
        query_condition = "user_name"
        delt_condition = "delt"

        # 构建 SQL 更新语句
        set_clause = ", ".join([f"{key} = %s" for key in new_data.keys()])  # 例如 "age = %s, address = %s"
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {query_condition} = %s AND {delt_condition} = 0"

        try:
            # 执行 SQL 语句
            values = list(new_data.values())  # 获取新数据的值
            values.append(user_name)  # 将用户名添加到参数列表的最后
            self.cursor.execute(sql, tuple(values))
            self.connection.commit()  # 提交事务
            return True
        except pymysql.MySQLError as e:
            print(f"更新数据失败: {e}")
            self.connection.rollback()  # 回滚事务
            return False

    def query_data(self, table_name, query_condition, query_value):
        sql = f"SELECT * FROM {table_name} WHERE {query_condition} = %s"
        try:
            self.cursor.execute(sql, (query_value,))
            results = self.cursor.fetchall()  # 获取所有查询结果
            return results
        except pymysql.MySQLError as e:
            print(f"查询数据失败: {e}")
            return None

    def delete_data(self, table_name, condition_column, condition_value):
        """实现软删除"""
        sql = f"UPDATE {table_name} SET delt = 1 WHERE {condition_column} = %s"
        try:
            self.cursor.execute(sql, (condition_value,))
            self.connection.commit()  # 提交事务
            return True
        except pymysql.MySQLError as e:
            print(f"逻辑删除数据失败: {e}")
            return False

    def get_all_user(self):
        try:
            self.cursor.execute("SELECT id, user_name, user_password, user_role FROM user_info WHERE delt = 0")
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            print(f"Error fetching records: {e}")
            return []

    def close(self):
        self.cursor.close()
        self.connection.close()


class PageManager(QMainWindow):

    def __init__(self):
        super().__init__()
        db_config = {"host": "localhost",
                     "user": "root",
                     "password": "root",
                     "database": "class_emotion"}
        # print("数据库连接成功")

        self.db_manager = DbManager(db_config)
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.setGeometry(200, 100, 1900, 800)  # 设置窗口位置和大小
        self.pages = {}
        self.currentUserType = None  # None: 无用户, 1: 管理员, 0: 普通用户
        self.init_pages()
        # self.stacked_widget.setStyleSheet("background: transparent;")
        # self.stacked_widget.setStyleSheet("""
        #     background-image: url(ui/background.png);
        # """)

    def init_pages(self):
        # 初始化页面并注册

        self.register_page("register", registerPageClass)
        self.register_page("logIn", logInPageClass)
        self.register_page("emotion_monitoring", EmotionMonitoringUI)  # 新增监控页面
        # 默认显示第一个页面
        self.switch_page("logIn")


    def register_page(self, name, page_class):
        if name not in self.pages:
            page = page_class(self.switch_page, self.db_manager) if name in ["register", "logIn", "adminmenu",
                                                                             "usermanage", "detect", "record",
                                                                             "userInfo"] else page_class(
                self.switch_page)
            self.pages[name] = page
            self.stacked_widget.addWidget(page)

    def switch_page(self, name):
        print(f"Attempting to switch to page: {name}")

        # 页面切换逻辑...
        page = self.pages.get(name)
        if page:
            self.stacked_widget.setCurrentWidget(page)


if __name__ == "__main__":
    app = QApplication([])
    window = PageManager()
    window.show()
    sys.exit(app.exec_())
