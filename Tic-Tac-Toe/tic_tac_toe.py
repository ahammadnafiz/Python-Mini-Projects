from tkinter import *
import random

class TicTacToe(Tk):
    def __init__(self):
        super().__init__()

        self.title('Tic-Tac-Toe')
        self.configure(bg='#F0F0F0')
        self.players = ['X', 'O']
        self.player = random.choice(self.players)
        self.label = Label(self,
                      text=f"{self.player}'s turn",
                      font=('consolas', 30),
                      bg='#F0F0F0',  
                      fg='#333333'
                      )
        self.label.pack(side=TOP, pady=10)
        
        self.restart()

        self.frame = Frame(self, bg='#F0F0F0')  
        self.frame.pack()

        self.buttons = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
                ]
        self.create_buttons()

    def create_buttons(self):
        for row in range(3):
            for col in range(3):
                self.buttons[row][col] = Button(self.frame,
                                          text="",
                                          font=('consolas', 40),
                                          width=5,
                                          height=2,
                                          command= lambda row=row, col=col: self.next_turn(row, col),
                                          padx=5,
                                          pady=5,
                                          bg='#CCCCCC',  
                                          fg='#333333', 
                                          activebackground='#AAAAAA', 
                                          activeforeground='#FFFFFF',  
                                          borderwidth=0,  
                                          relief="flat",
                                          )
                self.buttons[row][col].grid(row=row, column=col, padx=3, pady=3)

    def restart(self):
        restart_btn = Button(text='Restart',
                            font=('consolas', 20, 'bold'),
                             command=self.new_game,
                             bg='#4CAF50',  
                             fg='#FFFFFF',  
                             relief="flat", 
                             )
        restart_btn.pack(side=TOP, pady=10)

    def next_turn(self, row, col):
        if self.buttons[row][col]['text'] == "" and self.check_winner() is False:
            if self.player == self.players[0]:
                self.buttons[row][col]['text'] = self.player
                if self.check_winner() is False:
                    self.player = self.players[1]
                    self.label.config(text=f"{self.players[1]}'s turn")
                elif self.check_winner() is True:
                    self.label.config(text=f"{self.players[0]} wins")
                elif self.check_winner() == 'Tie':
                    self.label.config(text='Tie!')
            else:
                self.buttons[row][col]['text'] = self.player
                if self.check_winner() is False:
                    self.player = self.players[0]
                    self.label.config(text=f"{self.players[0]}'s turn")
                elif self.check_winner() is True:
                    self.label.config(text=f"{self.players[1]} wins")
                elif self.check_winner() == 'Tie':
                    self.label.config(text='Tie!')

    def check_winner(self):
        for row in range(3):
            if self.buttons[row][0]['text'] == self.buttons[row][1]['text'] == self.buttons[row][2]['text'] != "":
                self.buttons[row][0].config(bg='#4CAF50')  # Highlight winning line
                self.buttons[row][1].config(bg='#4CAF50')
                self.buttons[row][2].config(bg='#4CAF50')
                return True
        for col in range(3):
            if self.buttons[0][col]['text'] == self.buttons[1][col]['text'] == self.buttons[2][col]['text'] != "":
                self.buttons[0][col].config(bg='#4CAF50')
                self.buttons[1][col].config(bg='#4CAF50')
                self.buttons[2][col].config(bg='#4CAF50')
                return True
        if self.buttons[0][0]['text'] == self.buttons[1][1]['text'] == self.buttons[2][2]['text'] != "":
            self.buttons[0][0].config(bg='#4CAF50')
            self.buttons[1][1].config(bg='#4CAF50')
            self.buttons[2][2].config(bg='#4CAF50')
            return True
        elif self.buttons[0][2]['text'] == self.buttons[1][1]['text'] == self.buttons[2][0]['text'] != "":
            self.buttons[0][2].config(bg='#4CAF50')
            self.buttons[1][1].config(bg='#4CAF50')
            self.buttons[2][0].config(bg='#4CAF50')
            return True
        elif not self.empty_space():
            for row in range(3):
                for col in range(3):
                    self.buttons[row][col].config(bg='#FFC107')  
            return 'Tie'
        else:
            return False
    
    def empty_space(self):
        spaces = 9

        for row in range(3):
            for col in range(3):
                if self.buttons[row][col]['text'] != "":
                    spaces -= 1
        return spaces != 0

    def new_game(self):
        self.player = random.choice(self.players)
        self.label.config(text=f"{self.player}'s turn")

        for row in range(3):
            for col in range(3):
                self.buttons[row][col].config(text="", bg='#CCCCCC')
                self.buttons[row][col].config(state='normal')


if __name__ == '__main__':
    game = TicTacToe()
    game.mainloop()

