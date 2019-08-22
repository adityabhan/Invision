
# coding: utf-8

# In[98]:

import subprocess
import tkinter as tk
root = tk.Tk()
root.resizable(width=False, height=False)
root.title('InVision Enter Data')
rootHeight=190
rootWidth=500
root.geometry('{}x{}'.format(rootWidth,rootHeight))
#root.geometry('{}x{}'.format(460, 350))
root.configure(background='cyan')

def okButton(event):
    print("clicked!")
    a=""
    b=""
    #if entry1.get()
    a=entry1.get()
    b=entry2.get()
    print("first entry {}".format(a))
    print("second entry {}".format(b))
    try:
        if a:
        
            f= open("information.txt","w+")
            f.write("Name {}\r\n".format(a))
            #f.close()
        if b:
    
            f.write("Organization {}\r\n".format(b))
            #f.close()
            
    finally:
        print("closing file connection!")
        f.close()
        root.withdraw()
        subprocess.call(" python gui.py", shell=True)
        root.destroy()
    
    
    


# In[99]:


label1=tk.Label(root,text="Name",font=("Courier", 14))
label1.configure(background='cyan')
label2=tk.Label(root,text="Company/Organization Name",font=("Courier", 14))
label2.configure(background='cyan')

entry1=tk.Entry(root,width=25)
#entry1.configure()
entry1.configure(background='pink')
entry2=tk.Entry(root,width=25)
entry2.configure(background='pink')

label1.grid(row=0,pady=10,padx=10)
label2.grid(row=1,pady=10,padx=10)
entry1.grid(row=0,column=1,pady=10,padx=10)
entry2.grid(row=1,column=1,pady=10,padx=10)

ok_Button=tk.Button(root,text="OK")
img = tk.PhotoImage(file="ok-button-hi.png") # make sure to add "/" not "\"
ok_Button.config(image=img)
ok_Button.bind('<Button-1>',okButton)
ok_Button.grid(row=2,column=1,pady=10,padx=10)
#okButton.pack()


# In[100]:


root.mainloop()

