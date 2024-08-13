
def gpaResults (grade):
    if grade > 2.0:
        passDegree = True
    else:
        passDegree = False

    i=20
    return passDegree



gpa = input("What is your GPA? ")
i=10

if gpaResults(gpa):
    print "Your application was accepted."
else:
    print "You loser, get out of here."
print i