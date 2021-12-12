
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from statistics import mean
from matplotlib import style
style.use('ggplot')
import pandas as pd


# In[2]:


df = pd.read_csv('Bike Sharing hour.csv')
list(df)
df.drop(['dteday'],1,inplace = True)
df.drop(['instant'],1,inplace = True)
full_data = df.astype(float).values.tolist()


# In[3]:


header = list(df)


# In[4]:


xs = np.array(full_data)[:,:-1]
ys = np.array(full_data)[:,-1]


# In[5]:


x = xs[:,-2]
x


# In[6]:


plt.scatter(x,ys)


# In[7]:


def origin_shift(x,y):
    return x - np.mean(x) , y-np.mean(y)


# In[8]:


def pred_y(m,x):
    return m*x


# In[9]:


def calc_SSE(y,pred_y):
    return np.linalg.norm(y-pred_y)


# In[10]:


X,Y = origin_shift(x,ys)


# In[11]:


m = 0
y_hat = pred_y(m,X)


# In[12]:


y_hat
calc_SSE(Y,y_hat)


# In[41]:


alpha = 0.1
m = 0
m_list = []
prev_SSE = 99999999
curr_SSE = 0
i = 0
def reg_line(m,X,Y,m_list,prev_SSE,curr_SSE,alpha):
    if(alpha < 0.001):
        return m , m_list
    y_hat = pred_y(m,X)
    m_list.append(m)
    curr_SSE = calc_SSE(Y,y_hat)
    if(curr_SSE <= prev_SSE):
        prev_SSE = curr_SSE
        m+=alpha
        return reg_line(m,X,Y,m_list,prev_SSE,curr_SSE,alpha)
    else:
        m = m_list[-10]
        for i in range(10):
            m_list.remove(m_list[-1])
        prev_SSE = 999999
        alpha*=0.1
        m+=alpha
        return reg_line(m,X,Y,m_list,prev_SSE,curr_SSE,alpha)

m,m_list = reg_line(m,X,Y,m_list,prev_SSE,curr_SSE,alpha)


# In[37]:


m
c = np.mean(ys)
c


# In[38]:


m_list


# In[39]:


plt.scatter(x,ys)
plt.plot(x,m*x+c , color = 'g')


# In[40]:


fig = plt.figure()
ax = fig.gca()
t = []
t.append(x)
t.append(c)
t = tuple(t)
ax.scatter(x,ys)
plt.xlabel('casual')
plt.ylabel('total user count')
line, = ax.plot(x,(m*x)+c,color = 'g')
def update(m,x,c):
    line.set_data(x,m*x+c)
    return line

fig.canvas.draw()
ani = matplotlib.animation.FuncAnimation(fig, update, frames=m_list, blit=False, interval=10, fargs = t )
plt.show()

