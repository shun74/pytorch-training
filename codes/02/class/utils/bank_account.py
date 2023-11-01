class BankAccount:

  def __init__(self, name):
    self.name = name
    self.balance = 0
    self.interest_rate = 0.01

  def get_name(self):
    return self.name

  def get_balance(self):
    return self.balance
  
  def deposit(self, amount):
    self.balance += amount
  
  def withdraw(self, amount):
    self.balance -= amount

  def set_interest_rate(self, rate):
    self.interest_rate = rate

  def apply_interest(self):
    self.balance += self.balance * self.interest_rate
