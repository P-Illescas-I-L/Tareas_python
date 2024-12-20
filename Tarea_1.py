#Tarea No.1 MIT
'''Assignment:  
Write a program that does the following in order: 
1. Asks the user to enter a number “x” 
2. Asks the user to enter a number “y”  
3. Prints out number “x”, raised to the power “y”. 
4. Prints out the log (base 2) of “x”.  
'''
#libreria numpy para el logaritmo
import numpy as np
#Variables
num_1 = float(input("Introduce el primer número: "))
num_2 = float(input("Introduce el segundo número: "))

z = pow(num_1,num_2)
print("El número {num_1} elevado a la {num_2} es:",z)

w = np.log2(num_1)
print(w)