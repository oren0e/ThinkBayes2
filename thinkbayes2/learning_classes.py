# Python Object-Oriented Programming
# Lesson 1 #


# class Employee:
#     # initialize the class (constructor). The first argument is the instance itself (in each method). We call it self.
#     # method is a function that belongs to the class.
#     def __init__(self, first, last, pay):
#         self.first = first
#         # can also do self.fname = first (or whatever we like)
#         self.last = last
#         self.pay = pay
#         self.email = first + '.' + last + '@company.com'
#
#     def fullname(self):
#         return '{} {}'.format(self.first, self.last)
#
#
# emp_1 = Employee('Corey', 'Coen', '50000')
# emp_2 = Employee('Oren', 'Epshtain', '200000')
#
# print(emp_1.fullname())
#
# # The next 2 lines do exactly the same thing
# emp_1.fullname()  # here we dont need to pass anything because emp_1 passes itself in the form of 'self'
# print(Employee.fullname(emp_1))


# Lesson 2 #
# Class variables should be the same for each instance of the class


# class Employee:
#     num_of_emps = 0
#     raise_amount = 1.04  # class variable
#
#     # initialize the class (constructor). The first argument is the instance itself (in each method). We call it self.
#     # method is a function that belongs to the class.
#     def __init__(self, first, last, pay):
#         self.first = first
#         # can also do self.fname = first (or whatever we like)
#         self.last = last
#         self.pay = pay
#         self.email = first + '.' + last + '@company.com'
#
#         Employee.num_of_emps += 1  # increase value each time a new employee is created
#
#     def fullname(self):
#         return '{} {}'.format(self.first, self.last)
#
#     def apply_raise(self):
#         # we access the class variable by the instance. ('self' or class name)
#         self.pay = int(self.pay * self.raise_amount)
#
#
# emp_1 = Employee('Corey', 'Coen', 50000)
# emp_2 = Employee('Oren', 'Epshtain', 200000)
#
# # print(Employee.__dict__)
#
# # Employee.raise_amount = 1.05
# emp_1.raise_amount = 1.05  # only changes the class variable for that instance
# print(emp_1.__dict__)
#
# print(Employee.raise_amount)
# print(emp_1.raise_amount)  # first checks if the instance contains attribute, if not, then check if class does
# print(emp_2.raise_amount)
#
# # print(emp_1.pay)
# # emp_1.apply_raise()
# # print(emp_1.pay)
#
# print(Employee.num_of_emps)


# Lesson 3 #
# Regular methods, class methods, static methods

class Employee:
    num_of_emps = 0
    raise_amount = 1.04  # class variable

    # initialize the class (constructor). The first argument is the instance itself (in each method). We call it self.
    # method is a function that belongs to the class.
    def __init__(self, first, last, pay):
        self.first = first
        # can also do self.fname = first (or whatever we like)
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'

        Employee.num_of_emps += 1  # increase value each time a new employee is created

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        # we access the class variable by the instance. ('self' or class name)
        self.pay = int(self.pay * self.raise_amount)

    @classmethod  # denotes that this is a class method which gets the class
    # rather than the instance as before, as its first argument (cls instead of self)
    def set_raise_amt(cls, amount):
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)  # this line creates the new employee

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


emp_1 = Employee('Corey', 'Coen', 50000)
emp_2 = Employee('Oren', 'Epshtain', 200000)

# Change the raise amount to 5%
Employee.set_raise_amt(1.05)  # same effect as calling .raise_amount
# We can also call class methods on an instance but that does not make any sense.

print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

# Scenario: a string is received and the employee information is parsed from that
emp_str_1 = 'John-Doe-70000'
emp_str_2 = 'Steve-Smith-30000'
emp_str_3 = 'Jane-Doe-90000'

# first, last, pay = emp_str_1.split('-')

# new_emp_1 = Employee(first, last, pay)

# But we can also create in the class an alternative constructor
new_emp_1 = Employee.from_string(emp_str_1)

'''
Static methods does not pass anything, not the instance (like methods of the class) 
and not the class like class methods. A method should be static is 
when you don't access the instance or the class anywhere within the function (i.e if I dont use self or cls) 
'''
# Take a date and tell if that was a workday or not

import datetime
my_date = datetime.date(2016, 7, 10)

print(Employee.is_workday(my_date))


# Lesson 4
# Class inheritance

class Employee:

    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)


class Developer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)  # This is going to pass these 3 to Employee init method
        self.prog_lang = prog_lang


class Manager(Employee):

    def __init__(self, first, last, pay, employees = None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname())





''' We can initiate our subclasses with more information than our parent class can handle.
We do that by giving our subclass its own init method.
'''

dev_1 = Developer('Corey', 'Schafer', 50000, 'Python')
dev_2 = Developer('Test', 'Employee', 60000, 'Java')

mgr_1 = Manager('Sue', 'Smith', 90000, [dev_1])
print(mgr_1.email)

mgr_1.add_emp(dev_2)
mgr_1.remove_emp(dev_1)

mgr_1.print_emps()

print(dev_1.email)
print(dev_1.prog_lang)

print(help(Developer))

print(dev_1.pay)
dev_1.apply_raise()
print(dev_1.pay)

''' Python has 2 built-in functions isinstance and issubclass'''
print(isinstance(mgr_1, Manager))
print(isinstance(mgr_1, Employee))
print(isinstance(mgr_1, Developer))

print(issubclass(Manager, Employee))
print(issubclass(Developer, Employee))
print(issubclass(Manager, Developer))