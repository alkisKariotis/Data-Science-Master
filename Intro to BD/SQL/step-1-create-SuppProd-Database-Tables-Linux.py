#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 13:30:15 2018

@author: acggs
"""

import pymysql

# Open database connection
#mysql_config --socket  to get the socket


#Create a database for supplierProduct,delete it if it exists
# it deletes it
# Also create tables, which are empty initially. The tables will be 
# populated with data form the command prompt

# Open database connection
db = pymysql.connect(host="localhost",
                     user="ds",
                     passwd="0019cg!CG!",
                     unix_socket="/var/run/mysqld/mysqld.sock" )

cursor=db.cursor()

cursor.execute("DROP DATABASE IF EXISTS supplierProduct");


#create a database 
cursor.execute("create database supplierProduct")

#make it current
cursor.execute("use supplierProduct")



# Create supplier table
sql1 = """CREATE TABLE supplier (
         id  CHAR(20) NOT NULL primary key,
         name  CHAR(20),
         status INT,  
         city CHAR(20))"""

# Create product table
sql2 = """CREATE TABLE product (
         id  CHAR(20) NOT NULL primary key,
         name  CHAR(20),
         color CHAR(20),
         weight  float,  
         city CHAR(20))"""

#create supplierProduct table
sql3 = """CREATE TABLE supplierProduct2 (
         sid   CHAR(20), 
         foreign key (sid) references supplier(id),
         pid char(20), 
         qty INT,
         foreign key (pid) references product(id),
         primary key(sid,pid))"""
         

#this is where the queries are executed
cursor.execute(sql1)
cursor.execute(sql2)
cursor.execute(sql3)



# disconnect from server
db.close()
