from tkinter import *


class Calculator(Tk):
    def __init__(self):
        super().__init__()
        self.title('Calculator')
        self.geometry('490x620')
        
        self.entry = Entry(self, 
                           font=('Arial', 30))
        self.entry.grid(row=0, column=0, columnspan=4, padx=20, pady=30)

        button_text = ['7', '8', '9', '/', '4', '5', '6', '*', '1', '2', '3', '-', '0', 'C', '=', '+']
        row = 1
        column = 0

        for text in button_text:
            button = Button(self, 
                            text=text,
                            width=8,
                            height=4,
                            font=('Arial', 12, 'bold'),
                            command= lambda x = text: self.button_click(x))
            button.grid(row=row, column=column, padx=5, pady=5)
            column += 1
            if column > 3:
                column = 0
                row += 1
    
    def button_click(self, text):
        if text == 'C':
            self.entry.delete(0, END)
        elif text == '=':
            try:
                result = str(eval(self.entry.get()))
                self.entry.delete(0, END)
                self.entry.insert(END, result)
            except:
                self.entry.delete(0, END)
                self.entry.insert(END, 'Error')
        else:
            current = self.entry.get()
            self.entry.delete(0, END)
            self.entry.insert(END, f"{current}{text}")

if __name__ == '__main__':
    window = Calculator()
    window.mainloop()
