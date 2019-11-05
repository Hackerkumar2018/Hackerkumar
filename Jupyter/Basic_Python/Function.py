#!/usr/bin/env python
# coding: utf-8

# # Function

# In[2]:


def function():
    print("call, the function")
    


# In[7]:


#1. no arug no return type
def add():
    var1 = int(input("enter the variable 1 : "))
    var2 = int(input("enter the variable 2 : "))
    
    var3 = var1 + var2
    
    print("sum = ",var3)
    


# In[9]:


#with arug no return type
def sub(var1,var2):
    var3 = var1 - var2
    
    print("sub = ",var3)
    


# In[11]:


#no arug and with return type
def multiply():
    var1 = int(input("enter the num1 : "))
    var2 = int(input("enter the num2 : "))
    
    var3 = var1 * var2
    
    return var3


# In[14]:


#with argument and with return type
def div(var1,var2):
    var3 = var1/var2
    
    return var3

