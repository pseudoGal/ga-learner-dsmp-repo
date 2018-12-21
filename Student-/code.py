# --------------
# Code starts here


class_1 = ['Geoffrey Hinton','Andrew Ng','Sebastian Raschka','Yoshua Bengio']
class_2 = ['Hilary Mason','Carla Gentry','Corinna Cortes']

new_class = class_1 + class_2
print(new_class)
new_class.append('Peter Warden')
print(new_class)
new_class.remove('Carla Gentry')
print(new_class)
# Code ends here


# --------------
# Code starts here
courses = {'Math': 65,'English':70,'History':80,'French':70,'Science':60}

total = sum(courses.values())
#print (x)
#list(courses.values())
print("Total: ", total)
#for i in x:
#   x = x + 1
t1= (total/500)
percentage = t1*100
print("Percentage:", percentage)




# Code ends here


# --------------
# Code starts here

mathematics = {'Geoffrey Hinton':78,'Andrew Ng':95,'Sebastian Raschka':65,'Yoshua Benjio':50,'Hilary Mason':70,'Corinna Cortes':66,'Peter Warden':75 }

max_marks_scored = max(mathematics,key = mathematics.get)
#print (max_marks_scored)

topper = max_marks_scored
print(topper)
# Code ends here  


# --------------
# Given string
topper = "andrew ng"
print(topper.split(" "))

first_name = topper[0:6]
print(first_name)
last_name = topper[7:10]
print(last_name)
full_name = last_name +" "+ first_name
print("full_name:",full_name)
#full_name =  last_name+first_name 
#print(full_name)
certificate_name = full_name.upper()
print(certificate_name)
#print(topper.split(',', 2))

# Code starts here



# Code ends here


