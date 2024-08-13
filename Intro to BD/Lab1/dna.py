dna = str.lower(input("What is your DNA? (enter a sequence of a t c g) "))
countCG = 0
for x in dna:
	if x == "c" or x == "g":
		countCG = countCG + 1

percent = 100 * countCG / len(dna)
print ("Your DNA consists of", percent, "% C and G.")
