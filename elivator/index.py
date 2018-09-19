import elivator as e;
import random as r;
floors = [0,1,2,3,4];
Elivator = e.elivator;
elivator = Elivator();

for i in range(100):
    elivator.addPassenger(floors[r.randrange(len(floors))],2);

print(elivator.probArray());
print(elivator.dirsToGoAll());
