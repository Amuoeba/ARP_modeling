from __future__ import annotations
# Imports from external libraries
import sqlite3
import os
from typing import Tuple,Type
import audioread
import numpy as np
import random
# Imports from internal libraries
import config


class Table:
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def create_table(self)-> str:
        raise NotImplementedError

    @property
    def add_element(self)-> str:
        raise NotImplementedError

    def column_rename_map(self):
        raise NotImplementedError


class TABLE_train(Table):
    name = "train"
    create_table = f"""CREATE TABLE IF NOT EXISTS {name} (
                    idx INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                    speaker_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sample_rate INT NOT NULL,
                    data_size INT NOT NULL
                );"""

    add_element = f""" INSERT INTO {name}(speaker_id,video_id,file_name,file_path,sample_rate,data_size)
              VALUES(?,?,?,?,?,?) """

class TABLE_test(Table):
    name = "test"
    create_table = f"""CREATE TABLE IF NOT EXISTS {name} (
                    idx INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                    speaker_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sample_rate INT NOT NULL,
                    data_size INT NOT NULL
                );"""

    add_element = f""" INSERT INTO {name}(speaker_id,video_id,file_name,file_path,sample_rate,data_size)
              VALUES(?,?,?,?,?,?) """




class FileDatabase:

    def __init__(self, dbname):
        self.dbname = dbname
        self.connection = None

    def _check_db_existance_(self) -> bool:
        return os.path.exists(self.dbname)

    def _table_exists_(self, table:Table) -> bool:
        show_all_tables = """SELECT name FROM sqlite_master WHERE type='table';"""
        try:
            table_name = table.name
        except:
            if type(table) == str:
                table_name = table
            else:
                raise ValueError("Table should be of type Table or str")
        cursor = self.connection.cursor()
        cursor.execute(show_all_tables)
        results = [x[0] for x in cursor.fetchall()]
        if table_name in results:
            return True
        else:
            return False

    def _is_table_empty_(self,table:Table):
        cursor = self.connection.cursor()
        sq_count = f"""SELECT count(*) as total FROM {table.name}"""
        cursor.execute(sq_count)
        count = cursor.fetchone()[0]
        if count > 0:
            return False
        else:
            return True

    @staticmethod
    def read_files(path):
        for root,dirs,files in os.walk(path):
            rest,video_name = os.path.split(root)
            _,speaker_id = os.path.split(rest)
            for file in files:
                file_path = f"{root}/{file}"
                yield (speaker_id,video_name,file,file_path,0,0)

    def connect(self) -> FileDatabase:
        if self._check_db_existance_():
            print(f"Connecting to: {self.dbname} ")
        else:
            print(f"Database {self.dbname} doesn't exist yet... creating new database and connecting to it")

        self.connection = sqlite3.connect(self.dbname, check_same_thread=False)
        print(f"Connected to: {self.dbname}")
        return self

    def disconnect(self):
        if self.connection is not None:
            self.connection.close()
            print(f"Closing connection to: {self.dbname}")
        else:
            print("Connection nonexistent")
        return self

    def create_table(self, table:Table):
        if not self._table_exists_(table):
            cursor = self.connection.cursor()
            cursor.execute(table.create_table)
            self.connection.commit()
            print(f"Created table: {table.name}")
        else:
            print(f"Did nothing.Table {table.name} already exists")
        return self

    def drop_table(self,table:Table):
        print(f"Dropping table: {table.name}")
        cursor = self.connection.cursor()
        sq_drop = f"DROP TABLE IF EXISTS {table.name}"
        cursor.execute(sq_drop)
        self.connection.commit()
        return self


    def remove_table_rows(self,table:Table):
        print(f"Removing all rows from table: {table.name}")
        sq_delete = f"""DELETE FROM {table.name}"""
        cursor = self.connection.cursor()
        cursor.execute(sq_delete)
        self.connection.commit()
        return self

    def add_element(self, table: Table, values: Tuple):
        cursor = self.connection.cursor()
        cursor.execute(table.add_element, values)
        self.connection.commit()

    def populate_table(self, table:Table, data_root, owerwrite=False):
        counter = 0
        if self._is_table_empty_(table):
            print(f"Populating table: {table.name}")
            for file_attribute_tup in self.read_files(data_root):
                print(f"\r Adding element {counter}: {file_attribute_tup}", end="")
                self.add_element(table,file_attribute_tup)
                counter += 1
            return self
        elif owerwrite:
            print(f"Owervriting table: {table.name}")
            self.remove_table_rows(table)
            for file_attribute_tup in self.read_files(data_root):
                print(f"\r Adding element {counter}: {file_attribute_tup}",end="")
                self.add_element(table,file_attribute_tup)
                counter += 1
            return self
        else:
            print(f"Table: {table.name} already populated. Doing nothing")
            return self

    def get_num_elements(self,table:Type[Table]):
        cursor = self.connection.cursor()
        querry = f"SELECT COUNT(*) FROM {table.name}"
        cursor.execute(querry)
        return cursor.fetchone()[0]

    def get_nth_item(self,item, table: Type[Table]):
        cursor = self.connection.cursor()
        querry = f"SELECT * FROM {table.name} LIMIT 1 OFFSET {item}"
        cursor.execute(querry)
        return cursor.fetchone()

    def get_random_same(self,table:Type[Table],idx,id):
        cursor = self.connection.cursor()
        querry = f"SELECT * FROM {table.name} WHERE speaker_id = '{id}' AND idx != {idx}"
        cursor.execute(querry)
        pos__samples = cursor.fetchall()
        return random.choice(pos__samples)

    def get_random_different(self,table:Type[Table],id):
        cursor = self.connection.cursor()
        querry = f"SELECT * FROM {table.name} WHERE speaker_id != '{id}'"
        cursor.execute(querry)
        pos__samples = cursor.fetchall()
        return random.choice(pos__samples)






if __name__ == '__main__':
    print(f"Running database creation: {os.path.basename(__file__)}")

    file_db = FileDatabase(config.DATABASE)
    file_db.connect()
    # file_db.drop_table(TABLE_train)
    file_db.create_table(TABLE_train).populate_table(TABLE_train,config.TRAIN_DATA)
    file_db.create_table(TABLE_test).populate_table(TABLE_test, config.TEST_DATA)


    # for item in file_db.read_files(config.TRAIN_DATA):
    #     print(item)

