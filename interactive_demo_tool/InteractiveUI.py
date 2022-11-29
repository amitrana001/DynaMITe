# import tkinter as tk

# class Application(tk.Frame):
#     def __init__(self, master=None):
#         tk.Frame.__init__(self, master)
#         master.title("Test") #Controls the window title.
#         self.pack()
#         self.buttons = {}
#         self.prev_pressed = None
#         self.num_instances = 0
#         self.xPos = 0
#         self.yPos = 0
#         self.createWidgets()
        
#     def createWidgets(self):
#         floors = [i for i in range(5)]
#         # buttons = {}
#         # for floor in floors:
#         #     if(yPos == 5):
#         #         xPos = xPos + 1
#         #         yPos = 0
#         #     if(xPos == 8):
#         #         yPos = 2
#         #     self.num_instances+=1
#         #     self.buttons[floor] = tk.Button(self, width=3, text=floor, 
#         #                                     command = lambda f=floor: self.pressed(f))
#         #     self.buttons[floor].grid(row=xPos, column =yPos)
#         #     yPos = yPos +1
#         self.ADD = tk.Button(self, text="add instance", fg="green",
#                     command=self.add_button).grid(row = self.xPos, column = self.yPos)
#         self.REMOVE = tk.Button(self, text="remove instance", fg="green",
#                     command=self.remove_button).grid(row = self.xPos, column = self.yPos+1)
#         self.QUIT = tk.Button(self, text="QUIT", fg="red",
#                     command=root.destroy).grid(row = 0, column = 2)

#     def add_button(self):
#         self.xPos+=1
#         # self.yPos+=1
#         self.buttons[self.num_instances] = tk.Button(self, width=3, bg = "gray",text=self.num_instances, 
#                                             command = lambda f=self.num_instances: self.pressed(f))
#         self.buttons[self.num_instances].grid(row=self.xPos, column =self.yPos)
#         self.num_instances+=1
    
#     def remove_button(self):
#         self.remove_mode = not self.remove_mode

#     def pressed(self, index):
#         print("number pressed", index)
#         if self.prev_pressed is not None:
#             self.buttons[self.prev_pressed].configure(bg = "gray")
#         self.prev_pressed = index
#         self.buttons[index].configure(bg = "red")

# root = tk.Tk()
# app = Application(master=root)
# app.mainloop()
import tkinter as tk

USER_RGB_DICT = {}


def choose_color():
    color_code = tk.colorchooser.askcolor(title="Choose color")
    # we use unpacking here
    r, g, b = color_code[0]

    # store the choosings in the dict
    USER_RGB_DICT["R"] = r
    USER_RGB_DICT["G"] = g
    USER_RGB_DICT["B"] = b

    # example: another button for calculations is now available
    calc_button = tk.Button(root, text="Calculate with color", command=calculations_rgb)
    calc_button.pack()


def calculations_rgb():
    # since dicts preserve insertion order, we can do this
    r, g, b = USER_RGB_DICT.values()

    # some calculations
    if b > 200 and 50 > g and 50 > r:
        print("user really likes blue")


# initialize root
root = tk.Tk()
root.geometry("400x400")

# place a button
button = tk.Button(root, text="Select color", command=choose_color)
button.pack()

# fire the event loop
root.mainloop()