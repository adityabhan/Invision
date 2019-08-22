
# coding: utf-8

# In[22]:


#import os
import subprocess
import queue as qu

def Button1(event):
    #import imports
    t1 = threading.Thread(target=cam) 
    t1.start()
    #t1.join()
    #cam()
    print("1")
    
def Button2(event):
    #import imports
    t1 = threading.Thread(target=check) 
    t1.start()
    print("Use on a video")

def Button3(event):
    #import imports
    t1 = threading.Thread(target=check) 
    t1.start()
    
def Button4(event):
    #import imports
    t1 = threading.Thread(target=chooseFile(resultQueue)) 
    t1.start()
    #t1.join()
    #print(resultQueue.get(0))
    arg=resultQueue.get(0)
    subprocess.call(" python photoRec.py "+arg, shell=True)
    
    #resultQueue.empty..

def Button5(event):
    #import imports
    root.withdraw()
    subprocess.call(" python About.py", shell=True)
    #import opencamera
    print("Inside about!")
    root.deiconify()
    
def check():
    print("Inside check!")

def cam():
    #os.system("opencamera.py")
    root.withdraw()
    subprocess.call(" python opencamera.py", shell=True)
    #import opencamera
    print("Inside cam!")
    root.deiconify()
    
def chooseFile(queue):
    #os.system("opencamera.py")
    
    #subprocess.call(" python chooseFile.py", shell=True)
    #import opencamera
    print("choosing file!")
    from tkinter.filedialog import askopenfilename

    #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    queue.put(str(filename))
    


# In[37]:
dataForLabel=""
if __name__=='__main__':
    try:
        f=open("information.txt", "r")
        dataForLabel=f.read()
        print(dataForLabel)
    finally:
        print("data")
        f.close()

    
import threading 
import tkinter as tk

resultQueue=qu.Queue()

root = tk.Tk()
#root.resizable(width=False, height=False)
root.title('InVisio')
rootHeight=540
rootWidth=512
root.geometry('{}x{}'.format(rootWidth,rootHeight))
#root.geometry('{}x{}'.format(460, 350))

#printText="Your Organization is x"

leftframe = tk.Frame(root, bg='cyan', width=210, height=rootHeight, pady=10)
rightframe = tk.Frame(root, bg='white', width=450, height=450, pady=10)
bottomRight= tk.Frame(root, bg='white', width=310, height=320, pady=12,padx=-1)
#center = Frame(root, bg='gray2', width=50, height=40, padx=3, pady=3)
#btm_frame = Frame(root, bg='white', width=450, height=45, pady=3)
#btm_frame2 = Frame(root, bg='lavender', width=450, height=60, pady=3)

#root.grid_rowconfigure(0, weight=1)
#root.grid_columnconfigure(0, weight=1)

leftframe.grid(row=0,column=0,sticky="w")
rightframe.grid(row=0,column=1,sticky="n")
bottomRight.grid(row=0,column=1,sticky="s")

button_1=tk.Button(leftframe,bg='orange',text="Start surveillance",height =3, width =15 ,relief="groove",activebackground="pink",font=("Times", "12", "bold italic"))
button_2=tk.Button(leftframe,bg='orange',text="Use on a Video",height =3, width =15,activebackground="pink",font=("Times", "12", "bold italic"))
button_3=tk.Button(leftframe,bg='orange',text="Log File",height =3, width =15,activebackground="pink",font=("Times", "12", "bold italic"))
button_4=tk.Button(leftframe,bg='orange',text="Use on a Photo",height =3, width =15,activebackground="pink",font=("Times", "12", "bold italic"))
button_5=tk.Button(leftframe,bg='orange',text="About",height =3, width =15,activebackground="pink",font=("Times", "12", "bold italic"))

label=tk.Label(leftframe,bg='gray',fg='gold',text="INVISIO",height=4,width=13,font=("Courier", 17,'bold'))
label2=tk.Label(rightframe,bg='cyan',text=dataForLabel,height=8,width=25,font=("Courier", 15,'bold'))
#label3=tk.Label(rightframe)

#button_1.config( height = 20, width = 23 )
button_1.bind("<ButtonRelease-3> ",Button1)
button_2.bind("<Button-1>",Button2)
button_3.bind("<Button-1>",Button3)
button_4.bind("<Button-1>",Button4)
button_5.bind("<Button-1>",Button5)


label.grid(row=0,column=1,pady=5,padx=5)
label2.grid(row=0,column=1,pady=5,padx=5)
#label3.grid(row=1,column=1,pady=5,padx=5)
button_1.grid(row=1, column=1,pady=5,padx=10)
button_2.grid(row=2, column=1,pady=5,padx=5)
button_3.grid(row=3, column=1,pady=5,padx=5)
button_4.grid(row=4, column=1,pady=5,padx=5)
button_5.grid(row=5, column=1,pady=5,padx=5)
#button_1.pack()
#button_2.pack()
#button_3.pack()
#button_4.pack()
#button_5.pack()
#leftframe.pack()


# In[38]:


root.mainloop()

