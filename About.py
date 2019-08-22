
# coding: utf-8

# In[1]:


from datetime import datetime
import threading    
from tkinter import *
from tkinter import messagebox
import queue as qu
import time
import subprocess

def getTime():
    while True:
        date=datetime.now()
        #getTime()
        time.sleep(1)
        statusbar.config(text='date {} time {}'.format(date.strftime("%Y-%m-%d"),date.strftime("%H: %M: %S")))
        #sendTime.put(date)
        
def backButton(event):
    t1 = threading.Thread(target=callMain) 
    t1.start()
    
def callMain():
    root.withdraw()
    subprocess.call(" python gui.py", shell=True)

def checkClose():
    while True:
        root.protocol("WM_DELETE_WINDOW", on_closing)


def on_closing():
    var=messagebox.askokcancel("Quit", "Do you want to quit?")
    print("var is {}".format(var))
    if var:
        t1.killed=True
        root.destroy()


# In[2]:



root = Tk()
root.title('About InVision')
rootHeight=480
rootWidth=480
root.geometry('{}x{}'.format(rootHeight,rootWidth))

#root.resizable(width=False, height=False)
leftframe = Frame(root, bg='cyan', width=210, height=250, pady=10)

testString="InVisio has been designed and developed as a software which is capable of detecting and classifying objects, with focus on doing it under low-light or total dark conditions. Which will require combining object detection software with special hardware (IR camera) so as to capture the processable feed in dark. The primary intent of this project is to aid security surveillance, while it can be tailored for any application which involves detecting things and taking decision based on it. "
label=Label(leftframe,text=testString,font=("Courier", 15))
label.pack()
leftframe.pack()
#backButton=Button(root,text="Back",height =3, width =15,activebackground="red")
#backButton.bind("<Button-1>",backButton)
#backButton.pack()


statusbar=Label(root,relief=SUNKEN,anchor=W)
statusbar.pack(side=BOTTOM,fill=X)

#timeQueue=qu.Queue()

t1 = threading.Thread(target=getTime) 
#t1.start()


t2 = threading.Thread(target=checkClose) 
#t2.start()

#have to use a loop to update time
#while True:
#getTime(timeQueue)
#v=timeQueue.get(0).strftime("%Y-%m-%d %H:%M:%S")
#statusbar.config(text='date and time {}'.format(v))
#print(v)
    

#Status bar

root.mainloop()# main thread is busy here

print("outside mainloop!!")

#while t2.isAlive():
#    time.sleep(1)
#    print("main thread running!")


#t1.join()


# In[3]:


#import time
#time_string = time.strftime('%H:%M:%S')
#print(time_string)


# In[61]:


#import datetime
#now = datetime.datetime.now()
#print ("Current date and time : ")
#print (now.strftime("%Y-%m-%d %H:%M:%S"))

