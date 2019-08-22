
# coding: utf-8

# In[5]:


from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
TK().title('InVision')
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
return filename

