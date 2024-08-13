



#q1-b
#dna = str.lower(input("What is your DNA? "))
dna='aattccaacgttacgacd'
countCG = 0
countAT = 0
for x in dna:
    if x == "c" or x == "g":
        countCG = countCG + 1
    if x == "a" or x == "t":
        countAT += 1

percent = 100 * countCG / len(dna)
percentAT = 100 * countAT / len(dna)
print ("Your DNA consists of", percent, "% C and G.",percentAT, "% T and A")


i=0;
while i<len(dna):
    if i+3<=len(dna):
        print (dna[i:i+3])
    i=i+1

print("\n\n")



#q1-c
import numpy as np

#seq: dna sequence
#p: the probability to sample an element of the sequence
def sampleDna (seq, p):
    for i in dna:
        if np.random.rand()<=p:
           print(i,end=" ")



#q2-b
#find the product of number 1 till x
# essentially it dinds the number of 
# permutations 
def prod (x):
    pr=1
    while x>1:
        pr=pr*x
        x = x - 1
    return pr 

# some example to test it
print ("prod=",prod(3))

for i in range (1,11):
    print ('i=',i,' prod=',prod(i))



print("\n\n")
#q3-b
#prints the letter grade give a nymber grade
def getLetterGrade ( grade):
    if grade >= 70:
       return "A"
    elif 65 <= grade <= 69:
       return "A-"
    elif 60 <= grade <= 64:
       return "B+"
    elif 50 <= grade <= 59:
       return "B"
    elif 40 <= grade <= 49:
       return "C"
       


#gpa = float(input("What is your GPA? "))
gpa=12
print (getLetterGrade (gpa))




print("\n\n")
#q3-c
#reverses a list
def reverseList ( givenList):
    reverse=[]
   
    #range produces indexes from the end of the list till the first element
    #lists strart from 0 
    for i in range (len(givenList)-1,-1,-1):
        reverse.append(givenList[i])
    return reverse


print("\n\n")
#q3-d
#palindrome: checks if a string is a palindrome
#it returns true or false
def palindrome ( inpStr):
    i=0;
    j=len(inpStr)-1

    while i<j:
      if inpStr[i] != inpStr[j]:
         return False
      else:
         i = i + 1
         j = j - 1
    return True




print("\n\n")
#q4-a
##https://www.chegg.com/homework-help/questions-and-answers/python-code-translates-dna-protein-need-find-many-l-h-protein-sequence-need-change-add-cod-q32538399
def translate(seq): 
       
    table = { 
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W', 
    } 
    protein ="" 
    if len(seq)%3 == 0: 
        for i in range(0, len(seq), 3): 
            codon = seq[i:i + 3] 
            protein+= table[codon] 
    return protein 




print("\n\n")
#q4-b
keys = ['Ten', 'Twenty', 'Thirty']
values = [10, 20, 30]

sampleDict = dict(zip(keys, values))
print(sampleDict)



print("\n\n")
#q5-a
i = 0
while i <= 10:
    print(i)
    i += 1


print("\n\n")
#q5-b
print("Second Number Pattern ")
lastNumber = 6
for row in range(1, lastNumber):
    for column in range(1, row + 1):
        print(column, end=' ')
    print("")


print("\n\n")
#q5-c
def addSub (a, b):
    return a+b, a-b

add, sub = addSub (5, 2)
print(add)
print(sub)


print("\n\n")
#q6-a
def findDivisible(numberList):
    print("Given list is ", numberList)
    print("Divisible of 5 in a list")
    for num in numberList:
        if (num % 5 == 0):
            print(num)


numList = [10, 20, 33, 46, 55]
findDivisible(numList)


print("\n\n")
#q7-a
def printEvenIndex(str):
  for i in range(0, len(str), 2):
    print("idx[",i,"]", str[i] )
	
	
print("\n\n")	
#q7-b	
str1 = "Apple"
countDict = dict()
for char in str1:
  count = str1.count(char)
  countDict[char]=count
print(countDict)


print("\n\n")
#q7-c
str1 = "Aloha"
print("Original String is:", str1)

str1 = str1[::-1]
print("Reversed String is:", str1)


print("\n\n")
#q7-d
from string import punctuation

str1 = '/*Jon is @developer & musician!!'
print("The original string is : ", str1)

# Replace punctuations with #
replace_char = '#'

# Using string.punctuation to get the list of all punctuations
# use string function replace() to replace each punctuation with #

for char in punctuation:
    str1 = str1.replace(char, replace_char)

print("The strings after replacement : ", str1)






